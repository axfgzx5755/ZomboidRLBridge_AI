"""Run telemetry ticks against mock APIs, without controlling the game."""
import json
from pathlib import Path
import unittest

from lupa.lua51 import LuaRuntime
from test_lua_bridge import HARNESS


class TelemetryLuaTests(unittest.TestCase):
    def setUp(self):
        self.lua = LuaRuntime()
        self.lua.execute(HARNESS)
        self.lua.execute('''
            logs = {}
            print = function(message) table.insert(logs, message) end
            hp = 75
            player.getHealth = function() return hp end
            ZomboidRLNavigation = {scan=function() return '{"schema":2}' end}
        ''')
        root = Path(__file__).resolve().parents[1]
        self.lua.execute((root / 'ZomboidRLBridge/42/media/lua/client/Telemetry.lua').read_text())
        self.g = self.lua.globals()
        self.g.advance()

    def sample(self):
        self.g.advance()
        return json.loads(self.g.files['pz_state.txt'])

    def test_valid_sample(self):
        sample = self.sample()
        self.assertEqual(sample['x'], 10)
        self.assertEqual(sample['health'], 0.75)
        self.assertFalse(sample['invalidPlayerValues'])
        self.assertEqual(sample['navigation'], {'schema': 2})

    def test_invalid_numbers_are_flagged_and_recover(self):
        for field in ('px', 'py', 'pz', 'hp'):
            for value in ('nil', '"bad"', '0/0', 'math.huge', '-math.huge'):
                with self.subTest(field=field, value=value):
                    self.setUp()
                    self.lua.execute(f'{field} = {value}')
                    sample = self.sample()
                    self.assertTrue(sample['invalidPlayerValues'])
                    self.assertIsNone(sample['navigation'])
                    self.lua.execute('px, py, pz, hp = 10, 10, 0, 75')
                    self.assertFalse(self.sample()['invalidPlayerValues'])

    def test_missing_player_warning_is_throttled(self):
        self.lua.execute('player = nil')
        for _ in range(20):
            self.assertTrue(self.sample()['missingPlayer'])
        warnings = [v for v in self.g.logs.values() if 'No active player' in v]
        self.assertEqual(len(warnings), 1)


if __name__ == '__main__':
    unittest.main()
