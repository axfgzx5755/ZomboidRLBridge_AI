"""User-run room priority, incremental memory and Lua semantic wire regressions."""
import json
from pathlib import Path
import tempfile
import unittest
from lupa.lua51 import LuaRuntime
from local_explorer import Memory, Explorer
from navigation import Grid
from test_local_play import snapshot

class PlacesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.memory=Memory(Path(self.tmp.name)/'m.sqlite3','world')
    def tearDown(self):
        self.memory.close(); self.tmp.cleanup()
    def data(self):
        data=snapshot()
        for t in data['places']['tiles']:
            if (t['x'],t['y']) in ((0,0),(1,0)):
                t.update(building='building-a',room='room-a',name='hall')
        return data
    def test_new_room_priority_over_outside(self):
        data=self.data()
        for t in data['places']['tiles']:
            if (t['x'],t['y'])==(0,1):
                t.update(building='building-a',room='room-b',name='kitchen')
        action=Explorer(self.memory).plan(data)
        self.assertEqual(action['tile'],(0,1))
    def test_completion_reopens_when_new_tiles_are_seen(self):
        data=self.data(); grid=Grid.parse(data['map'],radius=4)
        self.memory.observe_places(data,grid)
        self.memory.visit((0,0)); self.memory.visit((1,0))
        self.assertEqual(self.memory.report()['buildings'][0]['status'],'known_area_covered')
        for t in data['places']['tiles']:
            if (t['x'],t['y'])==(2,0):
                t.update(building='building-a',room='room-a',name='hall')
        self.memory.observe_places(data,grid)
        self.assertEqual(self.memory.report()['buildings'][0]['status'],'incomplete')
    def test_closed_door_prevents_completion(self):
        data=self.data()
        data['doors']=[dict(x=0,y=0,z=0,north=True,open=False)]
        self.memory.observe_places(data,Grid.parse(data['map']))
        self.memory.visit((0,0)); self.memory.visit((1,0))
        self.assertEqual(self.memory.report()['buildings'][0]['status'],'incomplete')
        data['doors'][0]['open']=True
        self.memory.observe_places(data,Grid.parse(data['map']))
        self.assertEqual(self.memory.report()['buildings'][0]['status'],'known_area_covered')
    def test_door_state_refreshes_associations_outside_current_scan(self):
        data=self.data()
        for tile in data['places']['tiles']:
            if (tile['x'],tile['y'])==(0,-1):
                tile.update(building='building-b',room='room-b',name='hall')
        data['doors']=[dict(x=0,y=0,z=0,north=True,open=False)]
        grid=Grid.parse(data['map'])
        self.memory.observe_places(data,grid)
        for point in ((0,0),(1,0),(0,-1)):
            self.memory.visit(point)
        other=Memory(Path(self.tmp.name)/'m.sqlite3','other-world')
        try:
            other.observe_places(data,grid)
            other.visit((0,-1))
            # The door remains visible, but one building side is no longer loaded.
            data['places']['tiles']=[t for t in data['places']['tiles']
                                     if (t['x'],t['y'])!=(0,-1)]
            data['doors'][0]['open']=True
            self.memory.observe_places(data,grid)
            buildings=self.memory.report()['buildings']
            self.assertEqual(len(buildings),2)
            self.assertTrue(all(b['status']=='known_area_covered' for b in buildings))
            self.assertTrue(all(b['closed_known_doors']==1 for b in other.report()['buildings']))
            data['doors'][0]['open']=False
            self.memory.observe_places(data,grid)
            self.assertTrue(all(b['closed_known_doors']==1 for b in self.memory.report()['buildings']))
        finally:
            other.close()

    def test_unobserved_door_retains_last_known_state(self):
        data=self.data()
        data['doors']=[dict(x=0,y=0,z=0,north=True,open=False)]
        grid=Grid.parse(data['map'])
        self.memory.observe_places(data,grid)
        data['doors']=[]
        self.memory.observe_places(data,grid)
        self.assertEqual(self.memory.report()['buildings'][0]['closed_known_doors'],1)

    def test_actual_lua_labels_indoor_outdoor_unloaded(self):
        lua=LuaRuntime()
        lua.execute('''
        bdef={getX=function() return 100 end,getY=function() return 200 end}
        rdef={getID=function() return 7 end,getName=function() return 'kitchen' end}
        building={getDef=function() return bdef end}
        room={getRoomDef=function() return rdef end}
        function getCell() return {getGridSquare=function(self,x,y,z)
            if y~=0 or (x~=0 and x~=1) then return nil end
            return {getRoom=function() return x==0 and room or nil end,
                    getBuilding=function() return x==0 and building or nil end}
        end} end
        ''')
        path=Path(__file__).resolve().parents[1]/'ZomboidRLBridge/42/media/lua/client/Places.lua'
        lua.execute(path.read_text(encoding='utf-8'))
        result=json.loads(lua.globals().ZomboidRLPlaces.scan(0,0,0,1))
        self.assertEqual(len(result['tiles']),2)
        self.assertEqual(result['tiles'][0]['building'],'100:200')
        self.assertEqual(result['tiles'][0]['room'],'100:200:0:7')
        self.assertEqual(result['tiles'][1]['room'],'')

if __name__=='__main__': unittest.main()
