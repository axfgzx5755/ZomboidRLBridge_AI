"""User-run offline door regressions; no game process or keyboard input needed."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from lupa.lua51 import LuaRuntime

from door_env import DoorEnv, INTERACT, crosses_opening, load_scenario, scenario_from_sample


class DoorTests(unittest.TestCase):
    def setUp(self):
        self.door = dict(x=10,y=10,z=0,north=True)
        self.scenario = scenario_from_sample(SimpleNamespace(x=10.5,y=10.5),self.door)

    def test_crossing_requires_actual_opening_and_direction(self):
        self.assertTrue(crosses_opening([10.5,10.2],[10.5,9.8],self.door,1))
        self.assertFalse(crosses_opening([11.5,10.2],[11.5,9.8],self.door,1))
        self.assertFalse(crosses_opening([10.5,9.8],[10.5,10.2],self.door,1))
        west = dict(x=10,y=10,z=0,north=False)
        self.assertTrue(crosses_opening([9.8,10.5],[10.2,10.5],west,-1))

    def test_capture_roundtrip_and_invalid_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'door.json'
            path.write_text(json.dumps(self.scenario))
            self.assertEqual(load_scenario(path),self.scenario)
            self.scenario['target'] = [10.5,10.5]
            path.write_text(json.dumps(self.scenario))
            with self.assertRaises(ValueError): load_scenario(path)

    def sample(self,x=10.5,y=10.5):
        return SimpleNamespace(x=x,y=y,z=0,health=1)

    def test_open_reward_once_crossing_success_and_reset(self):
        backend = Mock()
        backend.reset.return_value = self.sample()
        backend.wait_for_update.return_value = self.sample()
        backend.target_is_free.return_value = True
        state = dict(open=False,locked=False,adjacent=True)
        def request(_,operation,door):
            if operation=='door_close': state['open']=False
            if operation=='door_open': state['open']=True
            return dict(state)
        with patch('navigation.LiveBackend',return_value=backend), \
             patch('door_env.door_request',side_effect=request), \
             patch.object(DoorEnv,'_local',return_value=np.zeros(125)):
            env = DoorEnv(self.scenario)
            obs,info = env.reset()
            reset_arena = backend.reset.call_args.args[0]
            self.assertEqual((reset_arena.x, reset_arena.y), (10.5, 10.5))
            self.assertEqual((env.arena.x, env.arena.y), (10, 10))
            self.assertEqual(obs.shape,(142,))
            self.assertTrue(env.observation_space.contains(obs))
            self.assertFalse(info['door_open'])
            _,reward,term,trunc,info = env.step(INTERACT)
            self.assertGreater(reward,1)
            self.assertTrue(info['opened_by_agent'])
            self.assertFalse(term or trunc)
            _,reward,_,_,_ = env.step(INTERACT)
            self.assertLess(reward,0)
            backend.wait_for_update.return_value = self.sample(10.5,9.5)
            _,_,term,_,info = env.step(6)
            self.assertTrue(info['crossed_door'])
            self.assertFalse(term)
            backend.wait_for_update.return_value = self.sample(10.5,8.5)
            _,_,term,_,info = env.step(6)
            self.assertTrue(term)
            self.assertTrue(info['is_success'])
            backend.wait_for_update.return_value = self.sample()
            _,info = env.reset()
            self.assertFalse(info['opened_by_agent'] or info['crossed_door'] or info['door_open'])
            env.close()
            backend.close.assert_called_once()

    def test_step_error_stops_input(self):
        backend = Mock()
        backend.reset.return_value = backend.wait_for_update.return_value = self.sample()
        with patch('navigation.LiveBackend',return_value=backend), \
             patch('door_env.door_request',return_value=dict(open=False,locked=False,adjacent=True)), \
             patch.object(DoorEnv,'_local',return_value=np.zeros(125)):
            env = DoorEnv(self.scenario)
            env.reset()
            backend.move.side_effect = RuntimeError('focus lost')
            with self.assertRaises(RuntimeError): env.step(1)
            self.assertTrue(env.done)
            backend.stop.assert_called()
            env.close()


class LuaDoorTests(unittest.TestCase):
    def setUp(self):
        self.lua = LuaRuntime()
        self.lua.execute('''
        opened=false; locked=false; px=10; py=10; toggles=0; files={}
        properties={}
        props={has=function(self,key) return properties[key] == true end}
        sprite={getProperties=function() return props end}
        square={getX=function() return 10 end,getY=function() return 10 end,getZ=function() return 0 end}
        door={getSquare=function() return square end,getNorth=function() return true end,
          getSprite=function() return sprite end,isDestroyed=function() return false end,
          isBarricaded=function() return false end,isLocked=function() return locked end,
          isLockedByKey=function() return false end,IsOpen=function() return opened end,
          ToggleDoor=function() opened=not opened; toggles=toggles+1 end}
        function square:getObjects() return {size=function() return 1 end,get=function() return door end} end
        player={getX=function() return px end,getY=function() return py end,getZ=function() return 0 end}
        function instanceof(o,t) return o==door and t=='IsoDoor' end
        function getCell() return {getGridSquare=function(self,x,y,z)
          if x==10 and y==10 and z==0 then return square end end} end
        function getFileWriter(path) return {write=function(self,text) files[path]=text end,close=function() end} end
        ''')
        source = Path(__file__).resolve().parents[1]/'ZomboidRLBridge/42/media/lua/client/DoorTraining.lua'
        self.lua.execute(source.read_text(encoding='utf-8'))

    def request(self,operation):
        self.lua.execute(f'ZomboidRLDoor.handle(player,"{operation}",10,10,0,1,"test")')
        return json.loads(self.lua.globals().files['pz_door_state.txt'])

    def test_open_idempotent_close_and_range(self):
        self.assertFalse(self.request('door_scan')['open'])
        self.assertTrue(self.request('door_open')['open'])
        self.assertTrue(self.request('door_open')['open'])
        self.assertEqual(self.lua.globals().toggles,1)
        self.assertFalse(self.request('door_close')['open'])
        self.lua.globals().px = 12
        self.assertFalse(self.request('door_open')['open'])

    def test_locked_door_rejected(self):
        self.lua.globals().locked = True
        with self.assertRaisesRegex(Exception, 'locked door'): self.request('door_open')
        self.assertEqual(self.lua.globals().toggles,0)

    def test_build42_property_api_and_door_type_rejections(self):
        # The mock deliberately exposes has(), not the removed Is() method.
        self.assertFalse(self.request('door_scan')['open'])
        for key, reason in (('DoubleDoor','double door'),('GarageDoor','garage door')):
            with self.subTest(property=key):
                self.lua.globals().properties[key] = True
                with self.assertRaisesRegex(Exception, reason):
                    self.request('door_open')
                self.lua.globals().properties[key] = False
        self.assertEqual(self.lua.globals().toggles,0)

    def test_scan_reports_nearby_door_and_required_tiles(self):
        self.lua.globals().px = 12
        with self.assertRaisesRegex(Exception, r'player tile=\(12,10,0\).*stand on \(10,10\) or \(10,9\)'):
            self.request('door_scan')
        self.assertEqual(self.lua.globals().toggles,0)


if __name__=='__main__':
    unittest.main()
