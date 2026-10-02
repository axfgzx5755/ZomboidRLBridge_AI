"""Offline planning and persistent experience; no API or game inputs here."""
from collections import deque
import json
import math
from pathlib import Path
import sqlite3
import time

from navigation import Grid


def door_id(door):
    return f"{door['x']}:{door['y']}:{door['z']}:{int(door['north'])}"


class Memory:
    def __init__(self,path,world):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(path)
        self.world=world
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS visits(world TEXT,x INT,y INT,n INT,PRIMARY KEY(world,x,y));
          CREATE TABLE IF NOT EXISTS failures(world TEXT,key TEXT,expires REAL,PRIMARY KEY(world,key));
          CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,world TEXT,created REAL,payload TEXT);
          CREATE TABLE IF NOT EXISTS places(world TEXT,x INT,y INT,building TEXT,room TEXT,name TEXT,walkable INT,
              PRIMARY KEY(world,x,y));
          CREATE TABLE IF NOT EXISTS portals(world TEXT,key TEXT,building TEXT,closed INT,
              PRIMARY KEY(world,key,building));
        ''')

    def visit(self,tile):
        self.db.execute('INSERT INTO visits VALUES(?,?,?,1) ON CONFLICT(world,x,y) DO UPDATE SET n=n+1',
                        (self.world,*tile))
        self.db.commit()

    def count(self,tile):
        row=self.db.execute('SELECT n FROM visits WHERE world=? AND x=? AND y=?',(self.world,*tile)).fetchone()
        return row[0] if row else 0

    def block(self,key,seconds=60):
        self.db.execute('INSERT OR REPLACE INTO failures VALUES(?,?,?)',(self.world,key,time.time()+seconds))
        self.db.commit()

    def blocked(self,key):
        row=self.db.execute('SELECT expires FROM failures WHERE world=? AND key=?',(self.world,key)).fetchone()
        return bool(row and row[0]>time.time())

    def record(self,event):
        self.db.execute('INSERT INTO events(world,created,payload) VALUES(?,?,?)',
                        (self.world,time.time(),json.dumps(event,allow_nan=False)))
        # Retain recent events per world; aggregated visitation survives pruning.
        self.db.execute('DELETE FROM events WHERE world=? AND id NOT IN (SELECT id FROM events WHERE world=? ORDER BY id DESC LIMIT 10000)',
                        (self.world,self.world))
        self.db.execute('DELETE FROM failures WHERE expires<?',(time.time(),))
        self.db.commit()

    def observe_places(self,snapshot,grid):
        places=snapshot.get('places')
        if places is None: return
        labels={(t['x'],t['y']):t for t in places['tiles']}
        self.db.executemany('INSERT OR REPLACE INTO places VALUES(?,?,?,?,?,?,?)',
            [(self.world,*point,t['building'],t['room'],t['name'],int(not grid.at(*point)[0]))
             for point,t in labels.items()])
        for door in snapshot['doors']:
            # A visible door's state applies to every remembered building side,
            # including sides whose place labels are outside the current scan.
            self.db.execute('UPDATE portals SET closed=? WHERE world=? AND key=?',
                            (int(not door['open']),self.world,door_id(door)))
            sides=[(door['x'],door['y']),
                   (door['x']-(not door['north']),door['y']-door['north'])]
            # Keep each known association even if one side is outside today's scan.
            for point in sides:
                t=labels.get(point)
                if t and t['building']:
                    self.db.execute('INSERT OR REPLACE INTO portals VALUES(?,?,?,?)',
                        (self.world,door_id(door),t['building'],int(not door['open'])))
        self.db.commit()

    def place_counts(self):
        rows=self.db.execute('''SELECT p.x,p.y,p.building,p.room,COALESCE(v.n,0)
            FROM places p LEFT JOIN visits v ON p.world=v.world AND p.x=v.x AND p.y=v.y
            WHERE p.world=?''',(self.world,)).fetchall()
        labels={}; rooms={}; buildings={}
        for x,y,b,r,n in rows:
            labels[(x,y)]=(b,r)
            if r: rooms[r]=rooms.get(r,0)+n
            if b: buildings[b]=buildings.get(b,0)+n
        return labels,rooms,buildings

    def report(self):
        rooms=self.db.execute('''SELECT p.building,p.room,MAX(p.name),COUNT(*),
            SUM(CASE WHEN COALESCE(v.n,0)>0 THEN 1 ELSE 0 END)
            FROM places p LEFT JOIN visits v ON p.world=v.world AND p.x=v.x AND p.y=v.y
            WHERE p.world=? AND p.room<>'' AND p.walkable=1 GROUP BY p.building,p.room''',
            (self.world,)).fetchall()
        result=[]
        for b,r,name,known,visited in rooms:
            result.append(dict(building=b,room=r,name=name,known_walkable_tiles=known,
                visited_tiles=visited,status='known_tiles_covered' if visited==known else
                'in_progress' if visited else 'unvisited'))
        buildings=[]
        for b in sorted({r['building'] for r in result if r['building']}):
            group=[r for r in result if r['building']==b]
            closed=self.db.execute('SELECT COUNT(*) FROM portals WHERE world=? AND building=? AND closed=1',
                (self.world,b)).fetchone()[0]
            buildings.append(dict(building=b,known_rooms=len(group),closed_known_doors=closed,
                status='known_area_covered' if not closed and all(r['status']=='known_tiles_covered' for r in group)
                else 'incomplete'))
        return dict(world=self.world,scope='observed ground-floor tiles only; not whole-building completion',
                    rooms=result,buildings=buildings)

    def close(self): self.db.close()


def edge_id(a,b): return f'edge:{a[0]},{a[1]}>{b[0]},{b[1]}'


class Explorer:
    def __init__(self,memory):
        self.memory=memory
        self.target=None
        self.last_tile=None
        self.last_action=None
        self.stalls=0

    def plan(self,snapshot):
        p=snapshot['player']; start=(math.floor(p['x']),math.floor(p['y']))
        if start!=self.last_tile:
            self.memory.visit(start); self.last_tile=start
        grid=Grid.parse(snapshot['map'],radius=4)
        if (grid.x,grid.y)!=start: raise ValueError('map/player position mismatch')
        self.memory.observe_places(snapshot,grid)
        labels,room_visits,building_visits=self.memory.place_counts()
        current_building=labels.get(start,('',''))[0]
        # Cardinal paths avoid diagonal corner cutting. Each action heads to a tile centre.
        paths={start:[]}; queue=deque([start])
        while queue:
            a=queue.popleft()
            for dx,dy in ((0,-1),(1,0),(0,1),(-1,0)):
                b=(a[0]+dx,a[1]+dy)
                if b not in paths and grid.edge(*a,dx,dy) and not self.memory.blocked(edge_id(a,b)):
                    paths[b]=paths[a]+[b]; queue.append(b)
        zombies=snapshot.get('zombies',[])
        if zombies:
            def danger(point):
                center=(point[0]+0.5,point[1]+0.5)
                return min(math.hypot(center[0]-z['x'],center[1]-z['y']) for z in zombies)
            if danger(start)<4.5:
                escape=[point for point in paths if point!=start]
                if escape:
                    destination=max(escape,key=lambda point:(
                        min(danger(tile) for tile in paths[point]),danger(point),-len(paths[point])))
                    self.target=None
                    return dict(kind='flee',tile=paths[destination][0],origin=start,
                                nearest_zombie_distance=danger(start))
        candidates=[]
        for door in snapshot['doors']:
            if door['open'] or door['locked'] or door['barricaded'] or not door['supported']: continue
            if self.memory.blocked('door:'+door_id(door)): continue
            sides=[(door['x'],door['y']),
                   (door['x']-(not door['north']),door['y']-door['north'])]
            if start in sides:
                self.target=None
                return dict(kind='open',door=door)
            for side in sides:
                if side in paths: candidates.append((len(paths[side]),side))
        if candidates:
            self.target=min(candidates)[1]
        elif self.target not in paths or self.target==start:
            points=[point for point in paths if point!=start]
            if not points: return dict(kind='wait',reason='no reachable route')
            def priority(point):
                building,room=labels.get(point,('',''))
                count=self.memory.count(point)
                if room and not room_visits.get(room,0): tier=0
                elif building and building==current_building and count==0: tier=1
                elif building and not building_visits.get(building,0): tier=2
                elif count==0: tier=3
                else: tier=4
                return tier,count,len(paths[point]) if tier<3 else -len(paths[point]),point
            self.target=min(points,key=priority)
        route=paths[self.target]
        return dict(kind='move',tile=route[0],origin=start)

    def feedback(self,action,before,after,outcome):
        if action['kind']=='open':
            if outcome not in ('opened','already_open'):
                self.memory.block('door:'+door_id(action['door']),120)
            self.stalls=0
        elif action['kind'] in ('move','flee'):
            goal=action['tile']; target=(goal[0]+0.5,goal[1]+0.5)
            old=math.hypot(before['x']-target[0],before['y']-target[1])
            new=math.hypot(after['x']-target[0],after['y']-target[1])
            self.stalls=self.stalls+1 if self.last_action==action and old-new<0.02 else 0
            self.last_action=action
            if self.stalls>=5:
                self.memory.block(edge_id(action['origin'],action['tile']))
                self.target=None; self.stalls=0
                outcome='route_blocked'
        self.memory.record(dict(action=action,before=before,after=after,outcome=outcome))
