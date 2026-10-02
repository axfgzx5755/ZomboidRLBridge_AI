require "Navigation"
-- Build 42.20 navigation arena. Only active while a Python lease is present.
local lastPoll = 0
local lastId = nil
local pending = nil
local cachedLease = {}
local function now() return getTimestampMs() / 1000 end

local function readLine(path)
    local opened, file = pcall(function() return getFileReader(path, false) end)
    if not opened or not file then return nil end
    local ok, line = pcall(function() return file:readLine() end)
    file:close()
    if ok then return line end
    return nil
end

local function words(line)
    local result = {}
    for word in string.gmatch(line or "", "%S+") do result[#result + 1] = word end
    return result
end

local function escape(value)
    return tostring(value):gsub("\\", "\\\\"):gsub('"', '\\"'):gsub("[%c]", " ")
end

local function ack(id, status, message)
    local file = getFileWriter("pz_training_ack.txt", true, false)
    if not file then error("cannot write training acknowledgement") end
    local ok, err = pcall(function()
        file:write(string.format('{"id":"%s","status":"%s","error":"%s"}\n',
            escape(id), status, escape(message or "")))
    end)
    file:close()
    if not ok then error(err) end
end

local function validLease(session)
    local line = readLine("pz_training_lease.txt")
    -- A Windows rename can briefly deny reads. Keep the last bounded lease
    -- during that race; Python explicitly writes expiry 0 on normal shutdown.
    if line then
        local lease = words(line)
        if #lease == 2 then cachedLease = lease end
    end
    local expires = tonumber(cachedLease[2]) or 0
    return cachedLease[1] == session and expires > now() and expires < now() + 60
end

local function restore(player)
    local data = player:getModData()
    local saved = data.ZomboidRLTraining
    if saved then
        player:setGodMod(saved.god)
        player:setGhostMode(saved.ghost)
        data.ZomboidRLTraining = nil
    end
end

local function resetStats(player)
    player:getBodyDamage():RestoreToFullHealth()
    local stats = player:getStats()
    for _, name in ipairs({"HUNGER", "THIRST", "FATIGUE", "PANIC", "STRESS", "PAIN"}) do
        stats:set(CharacterStat[name], 0)
    end
    stats:set(CharacterStat.ENDURANCE, 1)
end

local function freeTile(x, y, z)
    local square = getCell():getGridSquare(math.floor(x), math.floor(y), z)
    return square and square:TreatAsSolidFloor() and square:isFree(false)
end

local function validateArena(player, x, y, z, radius, localOnly)
    if player:isDead() then error("load a living character before training") end
    if player:getVehicle() then error("leave the vehicle before training") end
    if z ~= 0 or radius < 2 or radius > 20 then error("invalid arena dimensions") end
    if math.abs(player:getX() - x) > 50 or math.abs(player:getY() - y) > 50 then
        error("arena must be within 50 tiles of the loaded player")
    end
    if localOnly then
        if not freeTile(x, y, z) then error("reset tile is unloaded or obstructed") end
        return
    end
    for sx = math.floor(x - radius), math.floor(x + radius) do
        for sy = math.floor(y - radius), math.floor(y + radius) do
            local square = getCell():getGridSquare(sx, sy, z)
            if not square or not square:TreatAsSolidFloor() or not square:isFree(false) then
                error("arena is unloaded or obstructed at " .. sx .. "," .. sy .. "; choose open ground or a smaller radius")
            end
        end
    end
end

local function tick()
    local time = now()
    if time - lastPoll < 0.1 then return end
    lastPoll = time
    local player = getPlayer()
    if not player then return end
    -- Persisted flags are also restored after a save/reload if the lease expired.
    local saved = player:getModData().ZomboidRLTraining
    if saved and not validLease(saved.session) then
        restore(player)
        pending = nil
    end
    if pending then
        if time > pending.expires then
            ack(pending.id, "error", "teleport did not settle before timeout")
            restore(player)
            pending = nil
        elseif not pending.centered and
               ((math.floor(player:getX()) == math.floor(pending.x) and
                 math.floor(player:getY()) == math.floor(pending.y)) or
                (math.abs(player:getX() - math.floor(pending.x)) < 0.25 and
                 math.abs(player:getY() - math.floor(pending.y)) < 0.25)) and
               math.abs(player:getZ() - pending.z) < 0.1 and
               (pending.x ~= math.floor(pending.x) or pending.y ~= math.floor(pending.y)) then
            -- Lua may select the integer teleportTo overload. Apply fractional
            -- coordinates only after arrival (including small collision drift
            -- across the integer corner), then verify on the next tick.
            player:setX(pending.x)
            player:setY(pending.y)
            pending.centered = true
        elseif math.abs(player:getX() - pending.x) < 0.25 and
               math.abs(player:getY() - pending.y) < 0.25 and math.abs(player:getZ() - pending.z) < 0.1 then
            resetStats(player)
            ack(pending.id, "ok")
            pending = nil
        end
        return
    end
    local command = words(readLine("pz_training_command.txt"))
    if #command ~= 8 or command[1] == lastId then return end
    local id, session, expires, operation = command[1], command[2], tonumber(command[3]), command[4]
    lastId = id
    if not expires or expires <= time or expires > time + 30 or not validLease(session) then return end
    local ok, err = pcall(function()
        if isClient() or isServer() then error("training bridge supports single-player only") end
        local doorOperation = operation == "door_scan" or operation == "door_state" or operation == "door_open" or operation == "door_close"
        if not doorOperation and operation ~= "reset" and operation ~= "reset_local" and operation ~= "target" and operation ~= "map" then error("unknown operation") end
        local x, y, z, radius = tonumber(command[5]), tonumber(command[6]), tonumber(command[7]), tonumber(command[8])
        for _, value in ipairs({x, y, z, radius}) do
            if value ~= value or math.abs(value) == math.huge then error("non-finite coordinates") end
        end
        if not x or not y or not z or not radius then error("invalid coordinates") end
        if doorOperation then
            require "DoorTraining"
            if player:isDead() or player:getVehicle() then error("living player outside vehicle required") end
            if z ~= 0 or math.abs(player:getX()-x)>20 or math.abs(player:getY()-y)>20 then error("door outside training area") end
            if operation ~= "door_scan" and (radius ~= 0 and radius ~= 1) then error("invalid door orientation") end
            ZomboidRLDoor.handle(player,operation,x,y,z,radius,id)
            ack(id,"ok","")
            return
        end
        if operation == "map" then
            if radius ~= math.floor(radius) then error("map radius must be integer") end
            validateArena(player, x, y, z, radius, true)
            local json = ZomboidRLNavigation.scan(x,y,z,radius)
            local file = getFileWriter("pz_navigation_map.txt", true, false)
            if not file then error("cannot write navigation map") end
            local written, writeError = pcall(function()
                file:write('{"id":"' .. escape(id) .. '","map":' .. json .. '}')
            end)
            file:close()
            if not written then error(writeError) end
            ack(id, "ok", "")
            return
        end
        if operation == "target" then
            ack(id, "ok", freeTile(x, y, z) and "" or "blocked")
            return
        end
        validateArena(player, x, y, z, radius, operation == "reset_local")
        local data = player:getModData()
        if not data.ZomboidRLTraining then
            data.ZomboidRLTraining = {god=player:isGodMod(), ghost=player:isGhostMode(), session=session}
        elseif data.ZomboidRLTraining.session ~= session then
            error("another session owns the arena")
        end
        player:setGodMod(true)
        player:setGhostMode(true)
        resetStats(player)
        player:teleportTo(x, y, z)
        pending = {id=id, expires=expires, x=x, y=y, z=z}
    end)
    if not ok then
        restore(player)
        ack(id, "error", err)
    end
end

Events.OnTick.Add(function()
    local ok, err = pcall(tick)
    if not ok then
        local player = getPlayer()
        if player then pcall(function() restore(player) end) end
        pending = nil
        print("[ZomboidRLBridge] Training error: " .. tostring(err))
    end
end)
print("[ZomboidRLBridge] Training bridge ready (inactive without a Python lease)")
