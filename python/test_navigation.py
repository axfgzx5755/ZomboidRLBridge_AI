"""Offline contracts, reachability, collision and Lua geometry wire tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np
from gymnasium.utils.env_checker import check_env
from lupa.lua51 import LuaRuntime
from navigation import Grid, NavigationEnv, synthetic_grid
from navigation_train import evaluate, load_metadata, run
from telemetry import TelemetryError
from training_env import Arena


class NavigationTests(unittest.TestCase):
    def test_contract_and_seed(self):
        for stage in range(3):
            env=NavigationEnv(stage=stage)
            check_env(env,skip_render_check=True)
            first,info=env.reset(seed=321)
            second,again=env.reset(seed=321)
            np.testing.assert_array_equal(first,second)
            self.assertEqual(info,again)
            self.assertEqual(first.shape,(134,))
            point=tuple(map(int,env.target))
            self.assertIn(point,env.grid.reachable(*env.position))

    def test_edge_wall_and_corner_collision(self):
        g=synthetic_grid(Arena(),np.random.default_rng(0),0)
        g.cells[8,8,2]=1
        self.assertFalse(g.transition(0,0,1,0))
        self.assertFalse(g.transition(0,0,1,1))
        self.assertTrue(g.transition(0,0,0,1))
        g.cells[8,9,4]=1
        self.assertFalse(g.transition(1,0,0,0))

    def test_stall_and_success(self):
        env=NavigationEnv(stall_steps=3)
        env.reset(seed=42)
        for _ in range(3): obs,reward,term,trunc,info=env.step(0)
        self.assertFalse(term); self.assertTrue(trunc)
        self.assertEqual(info['terminal_reason'],'stalled')
        with self.assertRaises(RuntimeError): env.step(0)
        env.reset(seed=42)
        env.target=env.position+np.array([0.6,0])
        self.assertTrue(env.step(8)[2])

    def test_swept_collision(self):
        env=NavigationEnv(Arena(action_duration=1,allow_obstacles=True))
        env.reset(seed=42)
        env.grid.cells[8,8,2]=1
        env.step(8)
        self.assertLess(env.position[0],1)

    def test_unreachable_goal_excluded(self):
        g=synthetic_grid(Arena(),np.random.default_rng(0),0)
        g.cells[8,8,1:]=1
        self.assertEqual(g.reachable(0,0),{(0,0)})

    def test_missing_or_malformed_geometry(self):
        for data in (None,{},dict(schema=2,x=0,y=0,z=0,radius=2,cells=[0]*124),
                     dict(schema=2,x=0,y=0,z=0,radius=2,cells=[float('nan')]*125)):
            with self.assertRaises(TelemetryError): Grid.parse(data,radius=2)

    def test_evaluation_error_is_logged(self):
        env=Mock(stage=1)
        env.reset.side_effect=TelemetryError('paused')
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'result.jsonl'
            with self.assertRaises(TelemetryError): evaluate(Mock(),env,1,123,output,1)
            row=json.loads(output.read_text())
            self.assertEqual(row['terminal_reason'],'error')
            self.assertFalse(row['is_success'])
            env.stop.assert_called()

    def test_old_models_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'old.json'; path.write_text('{"schema":1}')
            with self.assertRaises(ValueError): load_metadata(path.with_suffix('.zip'))

    def test_lua_scan_wire_and_wall(self):
        lua=LuaRuntime()
        lua.execute('''
        function getCell() return {getGridSquare=function(self,x,y,z)
          if x==2 then return nil end
          return {TreatAsSolidFloor=function() return true end,
            isFree=function() return not (x==-1 and y==0) end,
            isBlockedTo=function(self,other) return x==0 and y==0 and other.x==1 end,
            isWindowTo=function() return false end,x=x,y=y}
        end} end
        ''')
        path=Path(__file__).resolve().parents[1]/'ZomboidRLBridge/42/media/lua/client/Navigation.lua'
        lua.execute(path.read_text())
        payload=json.loads(lua.globals().ZomboidRLNavigation.scan(0.3,0.4,0,2))
        grid=Grid.parse(payload,radius=2)
        self.assertEqual(len(payload['cells']),125)
        self.assertEqual(grid.at(-1,0)[0],1)
        self.assertEqual(grid.at(2,0)[0],1)
        self.assertFalse(grid.transition(0,0,1,0))
        self.assertTrue(grid.transition(0,0,0,1))

    def test_actual_telemetry_includes_navigation(self):
        lua=LuaRuntime()
        lua.execute("""
        package.preload["Navigation"]=function() end
        clock=1000
        function getTimestampMs() return clock end
        function getPlayer() return {getX=function() return 0.5 end,
          getY=function() return 0.5 end,getZ=function() return 0 end,
          getHealth=function() return 100 end} end
        function getCell() return {getGridSquare=function() return {
          TreatAsSolidFloor=function() return true end,isFree=function() return true end,
          isBlockedTo=function() return false end,isWindowTo=function() return false end
        } end} end
        function getFileWriter() return {write=function(self,s) result=s end,close=function() end} end
        Events={OnTick={Add=function(fn) tick=fn end}}
        """)
        root=Path(__file__).resolve().parents[1]/'ZomboidRLBridge/42/media/lua/client'
        lua.execute((root/'Navigation.lua').read_text())
        lua.execute((root/'Telemetry.lua').read_text())
        lua.execute('tick(); clock=1200; tick()')
        data=json.loads(lua.globals().result)
        self.assertEqual(data['health'],1)
        self.assertEqual(Grid.parse(data['navigation'],radius=2).x,0)

    def test_training_failure_saves_recoverable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'run'
            args=SimpleNamespace(resume=None,arena=None,live=False,transfer=False,stage=0,
                seed=42,output=output,cycles=1,steps=256,episodes=1,countdown=0,
                checkpoint_every=256,promote_at=0.8)
            with patch.object(NavigationEnv,'step',side_effect=TelemetryError('paused')):
                with self.assertRaises(TelemetryError): run(args)
            self.assertTrue((output/'interrupted.zip').exists())
            self.assertEqual(load_metadata(output/'interrupted.zip')['stage'],0)
            self.assertEqual(json.loads((output/'status.json').read_text())['error_type'],'TelemetryError')

    def test_map_request_ack_matches_written_snapshot(self):
        from test_lua_bridge import HARNESS
        lua=LuaRuntime(); lua.execute(HARNESS)
        root=Path(__file__).resolve().parents[1]/'ZomboidRLBridge/42/media/lua/client'
        lua.execute("""
        function getCell() return {getGridSquare=function() return {
            TreatAsSolidFloor=function() return true end,isFree=function() return true end,
            isBlockedTo=function() return false end,isWindowTo=function() return false end
        } end} end
        """)
        lua.execute((root/'Navigation.lua').read_text())
        lua.execute((root/'TrainingBridge.lua').read_text())
        lua.execute('files["pz_training_lease.txt"]="session 1030"; files["pz_training_command.txt"]="map1 session 1008 map 10 10 0 3"; advance()')
        ack=json.loads(lua.globals().files['pz_training_ack.txt'])
        data=json.loads(lua.globals().files['pz_navigation_map.txt'])
        self.assertEqual(ack['status'],'ok')
        self.assertEqual(ack['id'],data['id'])
        self.assertEqual(Grid.parse(data['map'],radius=3).x,10)
        self.assertFalse(lua.globals().god)

if __name__=='__main__': unittest.main()
