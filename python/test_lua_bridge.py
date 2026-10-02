"""Execute the real Lua bridge with mocked Build 42 APIs (requires lupa)."""

import json
from pathlib import Path
import unittest
from lupa.lua51 import LuaRuntime


HARNESS = r'''
package.preload["Navigation"] = function() end
files = {}
clock = 1000000
blocked = false
multiplayer = false
dead = false
god = false
ghost = true
modData = {}
px, py, pz = 10, 10, 0
CharacterStat = {HUNGER="hunger", THIRST="thirst", FATIGUE="fatigue", PANIC="panic", STRESS="stress", PAIN="pain", ENDURANCE="endurance"}
stats = {set=function(self, key, value) self[key] = value end}
player = {
 getX=function() return px end, getY=function() return py end, getZ=function() return pz end,
 getModData=function() return modData end,
 isDead=function() return dead end, getVehicle=function() return nil end,
 isGodMod=function() return god end, setGodMod=function(self, value) god=value end,
 isGhostMode=function() return ghost end, setGhostMode=function(self, value) ghost=value end,
 getStats=function() return stats end,
 getBodyDamage=function() return {RestoreToFullHealth=function() healed=true end} end,
 teleportTo=function(self,x,y,z) px,py,pz=math.floor(x),math.floor(y),z end,
 setX=function(self,x) px=x end, setY=function(self,y) py=y end,
}
function getPlayer() return player end
function getTimestampMs() return clock end
function isClient() return multiplayer end
function isServer() return false end
function getCell() return {getGridSquare=function() return {
 TreatAsSolidFloor=function() return true end, isFree=function() return not blocked end
} end} end
function getFileReader(path)
 if not files[path] then return nil end
 return {readLine=function() return files[path] end, close=function() end}
end
function getFileWriter(path)
 return {write=function(self, value) files[path]=value end, close=function() end}
end
Events = {OnTick={Add=function(callback) tick=callback end}}
function advance() clock=clock+200; tick() end
'''


class LuaTests(unittest.TestCase):
    def setUp(self):
        self.lua = LuaRuntime()
        self.lua.execute(HARNESS)
        source = Path(__file__).resolve().parents[1] / "ZomboidRLBridge/42/media/lua/client/TrainingBridge.lua"
        self.lua.execute(source.read_text())
        self.globals = self.lua.globals()

    def request(self, request_id="one", expires=1008):
        self.globals.files["pz_training_lease.txt"] = "session 1030"
        self.globals.files["pz_training_command.txt"] = f"{request_id} session {expires} reset 12 12 0 3"
        self.globals.advance()
        self.globals.advance()

    def ack(self):
        return json.loads(self.globals.files["pz_training_ack.txt"])

    def test_reset_and_cleanup(self):
        self.request()
        self.assertEqual(self.ack()["status"], "ok")
        self.assertEqual(self.globals.px, 12)
        self.assertTrue(self.globals.god)
        self.assertEqual(self.globals.stats.endurance, 1)
        self.globals.files["pz_training_lease.txt"] = "session 0"
        self.globals.advance()
        self.assertFalse(self.globals.god)
        self.assertTrue(self.globals.ghost)

    def test_rejections(self):
        for flag in ("blocked", "multiplayer", "dead"):
            with self.subTest(flag=flag):
                self.setUp()
                self.globals[flag] = True
                self.request()
                self.assertEqual(self.ack()["status"], "error")
                self.assertFalse(self.globals.god)

    def test_local_reset_allows_surrounding_obstacles(self):
        self.lua.execute("""
        function getCell() return {getGridSquare=function(self,x,y,z) return {
          TreatAsSolidFloor=function() return true end,
          isFree=function() return x == 12 and y == 12 end
        } end} end
        """)
        self.request()
        self.assertEqual(self.ack()["status"], "error")
        self.globals.files["pz_training_command.txt"] = "local session 1008 reset_local 12 12 0 3"
        self.globals.advance()
        self.globals.advance()
        self.assertEqual(self.ack()["status"], "ok")

    def test_fractional_reset_after_integer_teleport(self):
        self.globals.files["pz_training_lease.txt"] = "session 1030"
        self.globals.files["pz_training_command.txt"] = "center session 1008 reset_local 12.5 12.5 0 3"
        self.globals.advance()
        self.assertEqual(self.globals.px, 12)
        self.globals.advance()
        self.assertIsNone(self.globals.files["pz_training_ack.txt"])
        self.assertEqual((self.globals.px, self.globals.py), (12.5, 12.5))
        self.globals.advance()
        self.assertEqual(self.ack()["status"], "ok")

    def test_fractional_reset_with_collision_drift_across_tile_edge(self):
        for dx, dy in ((0.030518, -0.034668), (-0.03, 0.04), (-0.03, -0.04)):
            with self.subTest(dx=dx, dy=dy):
                self.setUp()
                self.globals.files["pz_training_lease.txt"] = "session 1030"
                self.globals.files["pz_training_command.txt"] = "drift session 1008 reset_local 12.5 12.5 0 3"
                self.globals.advance()
                self.globals.px, self.globals.py = 12 + dx, 12 + dy
                self.globals.advance()
                self.assertIsNone(self.globals.files["pz_training_ack.txt"])
                self.assertEqual((self.globals.px, self.globals.py), (12.5, 12.5))
                self.globals.advance()
                self.assertEqual(self.ack()["status"], "ok")

    def test_fractional_reset_does_not_accept_distant_position(self):
        self.globals.files["pz_training_lease.txt"] = "session 1030"
        self.globals.files["pz_training_command.txt"] = "far session 1008 reset_local 12.5 12.5 0 3"
        self.globals.advance()
        self.globals.px, self.globals.py = 11.5, 12
        self.globals.advance()
        self.assertEqual(self.globals.px, 11.5)
        self.assertIsNone(self.globals.files["pz_training_ack.txt"])
        self.globals.clock = 1009000
        self.globals.advance()
        self.assertEqual(self.ack()["status"], "error")

    def test_local_reset_rejects_blocked_origin(self):
        self.globals.blocked = True
        self.globals.files["pz_training_lease.txt"] = "session 1030"
        self.globals.files["pz_training_command.txt"] = "local session 1008 reset_local 12 12 0 3"
        self.globals.advance()
        self.assertEqual(self.ack()["status"], "error")
        self.assertEqual(self.globals.px, 10)

    def test_target_check_does_not_teleport(self):
        self.globals.files["pz_training_lease.txt"] = "session 1030"
        for blocked in (False, True):
            self.globals.blocked = blocked
            self.globals.files["pz_training_command.txt"] = f"target{blocked} session 1008 target 12 12 0 3"
            self.globals.advance()
            self.assertEqual(self.ack()["status"], "ok")
            self.assertEqual(self.ack()["error"], "blocked" if blocked else "")
            self.assertEqual(self.globals.px, 10)
            self.assertFalse(self.globals.god)

    def test_expired_command_ignored(self):
        self.request(expires=999)
        self.assertIsNone(self.globals.files["pz_training_ack.txt"])
        self.assertEqual(self.globals.px, 10)

    def test_lease_expiration_and_duplicate(self):
        self.request()
        self.globals.px = 13
        self.globals.advance()
        self.assertEqual(self.globals.px, 13)
        self.globals.clock = 1040000
        self.globals.advance()
        self.assertFalse(self.globals.god)

    def test_teleport_timeout_restores_flags(self):
        self.lua.execute("player.teleportTo = function() end")
        self.request()
        self.assertIsNone(self.globals.files["pz_training_ack.txt"])
        self.globals.clock = 1009000
        self.globals.advance()
        self.assertEqual(self.ack()["status"], "error")
        self.assertFalse(self.globals.god)
        self.assertTrue(self.globals.ghost)

    def test_api_failure_restores_flags(self):
        self.lua.execute('player.teleportTo = function() error("API failed") end')
        self.request()
        self.assertEqual(self.ack()["status"], "error")
        self.assertFalse(self.globals.god)

    def test_telemetry_files_compile(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / "ZomboidRLBridge/42/media/lua/client/Telemetry.lua",
                     root.parent / "mods/ZomboidRLBridge/42/media/lua/client/Telemetry.lua"):
            self.lua.execute("assert(loadstring(...))", path.read_text())

    def test_transient_lease_read_failure(self):
        self.request()
        self.globals.files["pz_training_lease.txt"] = None
        self.globals.advance()
        self.assertTrue(self.globals.god)
        self.globals.clock = 1040000
        self.globals.advance()
        self.assertFalse(self.globals.god)


if __name__ == "__main__":
    unittest.main()
