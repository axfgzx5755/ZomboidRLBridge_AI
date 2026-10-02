"""Offline local-play checks. No keyboard input, game process or API."""
import json
from pathlib import Path
import tempfile
import unittest
from lupa.lua51 import LuaRuntime
from local_explorer import Explorer, Memory, door_id, edge_id
from local_play import movement_towards, validate_snapshot
from test_lua_bridge import HARNESS

def snapshot():
    return dict(schema=1,player=dict(x=0.5,y=0.5,z=0,health=1,dead=False,vehicle=False),
                map=dict(schema=2,x=0,y=0,z=0,radius=4,cells=[0]*405),doors=[],
                places=dict(schema=1,tiles=[dict(x=x,y=y,building='',room='',name='')
                    for y in range(-4,5) for x in range(-4,5)]))

class ExplorerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'memory.sqlite3'
        self.memory=Memory(self.path,'save-a')
    def tearDown(self):
        self.memory.close(); self.temp.cleanup()
    def test_persistence_and_world_separation(self):
        self.memory.visit((2,3)); self.memory.block('door:test')
        other=Memory(self.path,'save-b')
        try:
            self.assertEqual(other.count((2,3)),0)
            self.assertFalse(other.blocked('door:test'))
        finally: other.close()
        again=Memory(self.path,'save-a')
        try:
            self.assertEqual(again.count((2,3)),1)
            self.assertTrue(again.blocked('door:test'))
        finally: again.close()
    def test_plan_and_failed_route_cooldown(self):
        planner=Explorer(self.memory); data=snapshot()
        validate_snapshot(data)
        action=planner.plan(data)
        self.assertEqual(action['kind'],'move')
        self.assertEqual(sum(abs(v) for v in action['tile']),1)
        for _ in range(7): planner.feedback(action,data['player'],data['player'],'observed')
        self.assertTrue(self.memory.blocked(edge_id(action['origin'],action['tile'])))
        self.assertNotEqual(planner.plan(data)['tile'],action['tile'])
    def test_door_selection_and_failure(self):
        data=snapshot()
        d=dict(x=0,y=0,z=0,north=True,open=False,locked=False,barricaded=False,supported=True)
        data['doors']=[d]; planner=Explorer(self.memory)
        action=planner.plan(data)
        self.assertEqual(action['kind'],'open')
        planner.feedback(action,data['player'],data['player'],'locked')
        self.assertTrue(self.memory.blocked('door:'+door_id(d)))
        self.assertEqual(planner.plan(data)['kind'],'move')
    def test_blocked_grid_waits(self):
        data=snapshot(); data['map']['cells']=[1]*405
        self.assertEqual(Explorer(self.memory).plan(data)['kind'],'wait')
    def test_reject_bad_geometry_and_dead_player(self):
        data=snapshot(); data['map']['x']=2
        with self.assertRaises(ValueError): validate_snapshot(data)
        data=snapshot(); data['player']['dead']=True
        with self.assertRaises(RuntimeError): validate_snapshot(data)
    def test_isometric_cardinal_mapping(self):
        p=dict(x=0.5,y=0.5)
        for target,expected in [((1,0),8),((-1,0),5),((0,1),7),((0,-1),6)]:
            action,duration=movement_towards(p,target)
            self.assertEqual(action,expected)
            self.assertLessEqual(duration,0.15)

class FreePlayLuaTests(unittest.TestCase):
    def setUp(self):
        self.lua=LuaRuntime(); self.lua.execute(HARNESS)
        self.lua.execute('''
        package.preload["Places"]=function() end
        ZomboidRLPlaces={scan=function() return '{"schema":1,"tiles":[]}' end}
        function player:getHealth() return 100 end
        ZomboidRLNavigation={scan=function() return '{"schema":2}' end}
        open=false
        local props={has=function() return false end}
        door={getSprite=function() return {getProperties=function() return props end} end,
            isDestroyed=function() return false end,isLocked=function() return false end,
            isLockedByKey=function() return false end,isBarricaded=function() return false end,
            getNorth=function() return true end,IsOpen=function() return open end,
            ToggleDoor=function() open=not open end}
        function instanceof(obj,name) return obj==door and name=='IsoDoor' end
        function getCell() return {getZombieList=function() return {size=function() return 0 end} end,
          getGridSquare=function(self,x,y,z)
            return {getObjects=function() return {
                size=function() return (x==10 and y==10) and 1 or 0 end,
                get=function() return door end} end}
        end} end
        ''')
        path=Path(__file__).resolve().parents[1]/'ZomboidRLBridge/42/media/lua/client/ActionBridge.lua'
        self.lua.execute(path.read_text(encoding='utf-8'))
    def request(self,operation='observe',session='session'):
        g=self.lua.globals()
        g.files['pz_training_lease.txt']='session 1030'
        g.files['pz_action_command.txt']=f'one {session} 1008 {operation} 10 10 0 1'
        g.advance()
        result=g.files['pz_action_ack.txt']
        return json.loads(result) if result else None
    def test_observe_does_not_reset_or_heal(self):
        result=self.request()
        self.assertEqual(result['status'],'ok')
        self.assertEqual(result['player']['health'],1)
        g=self.lua.globals()
        self.assertEqual((g.px,g.py),(10,10))
        self.assertFalse(g.god)
        self.assertIsNone(g.healed)
    def test_open_reports_actual_result(self):
        result=self.request('open')
        self.assertEqual(result['outcome'],'opened')
        self.assertTrue(result['doors'][0]['open'])
        self.assertFalse(self.lua.globals().god)
    def test_foreign_lease_cannot_execute(self):
        self.assertIsNone(self.request('open','other'))
        self.assertFalse(self.lua.globals().open)
    def test_reset_is_not_a_free_play_action(self):
        result=self.request('reset')
        self.assertEqual(result['status'],'error')
        self.assertIsNone(self.lua.globals().healed)

if __name__=='__main__': unittest.main()
