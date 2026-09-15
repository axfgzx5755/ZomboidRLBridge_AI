local LOG_INTERVAL_SECONDS = 0.5
local lastLogTime = 0.0
local lastMissingPlayerLogTime = 0.0
local lastInvalidPayloadLogTime = 0.0

-- getFileWriter resolves paths relative to the Zomboid user Lua directory.
local STATE_FILE_PATH = "pz_state.txt"
local hasLoggedWriterReady = false

local function nowSeconds()
    if getTimestampMs then
        return getTimestampMs() / 1000.0
    end
    if os and os.time then
        return os.time() + (os.clock() % 1.0)
    end
    return 0.0
end

local function safeGetPlayer()
    if getPlayer then
        local player = getPlayer()
        if player ~= nil then
            return player
        end
    end
    return nil
end

local function normalizeHealth(rawHealth)
    if rawHealth == nil then
        return nil
    end
    local health = tonumber(rawHealth)
    if health == nil then
        return nil
    end
    if health > 1.0 then
        health = health / 100.0
    end
    return math.max(0.0, math.min(1.0, health))
end

local function getPlayerCoords(player)
    if player == nil then
        return nil, nil, nil
    end
    local x = player.getX and player:getX() or nil
    local y = player.getY and player:getY() or nil
    local z = player.getZ and player:getZ() or nil
    return x, y, z
end

local function getPlayerHealth(player)
    if player == nil then
        return nil
    end
    local health = player.getHealth and player:getHealth() or nil
    if health == nil and player.getBodyDamage then
        local bodyDamage = player:getBodyDamage()
        if bodyDamage ~= nil and bodyDamage.getHealth then
            health = bodyDamage:getHealth()
        end
    end
    return normalizeHealth(health)
end

local function buildTelemetryPayload()
    local player = safeGetPlayer()
    if player == nil then
        print("[ZomboidRLBridge] buildTelemetryPayload: getPlayer() returned nil")
        return {
            x = 0.0, y = 0.0, z = 0.0, health = 1.0,
            missingPlayer = true, invalidPlayerValues = false,
        }
    end

    local x, y, z = getPlayerCoords(player)
    local health = getPlayerHealth(player)
    if x == nil or y == nil or z == nil or health == nil then
        print(string.format(
            "[ZomboidRLBridge] buildTelemetryPayload: player present but invalid values x=%s y=%s z=%s health=%s",
            tostring(x), tostring(y), tostring(z), tostring(health)
        ))
        return {
            x = 0.0, y = 0.0, z = 0.0, health = 1.0,
            missingPlayer = false, invalidPlayerValues = true,
        }
    end

    return {
        x = tonumber(x) or 0.0,
        y = tonumber(y) or 0.0,
        z = tonumber(z) or 0.0,
        health = health,
        missingPlayer = false,
        invalidPlayerValues = false,
    }
end

local function writeTelemetryFile(payload)
    if payload == nil then
        print("[ZomboidRLBridge] writeTelemetryFile: payload is nil")
        return
    end
    if getFileWriter == nil then
        print("[ZomboidRLBridge] writeTelemetryFile: getFileWriter is unavailable")
        return
    end

    local openOk, fileOrError = pcall(function()
        return getFileWriter(STATE_FILE_PATH, true, false)
    end)
    if not openOk then
        print("[ZomboidRLBridge] Failed to open state file: " .. tostring(fileOrError))
        return
    end

    local file = fileOrError
    if file == nil then
        print("[ZomboidRLBridge] getFileWriter returned nil for: " .. tostring(STATE_FILE_PATH))
        return
    end
    if not hasLoggedWriterReady then
        print("[ZomboidRLBridge] State writer opened: " .. tostring(STATE_FILE_PATH))
        hasLoggedWriterReady = true
    end

    local json = string.format(
        '{"x":%.6f,"y":%.6f,"z":%.6f,"health":%.6f}\n',
        payload.x, payload.y, payload.z, payload.health
    )
    local ok, writeErr = pcall(function()
        file:write(json)
        file:close()
    end)
    if not ok then
        print("[ZomboidRLBridge] Failed to write state file: " .. tostring(writeErr))
        pcall(function() file:close() end)
    end
end

local function logTelemetry()
    local payload = buildTelemetryPayload()
    local now = nowSeconds()
    if payload.missingPlayer then
        if (now - lastMissingPlayerLogTime) >= 10.0 then
            print("[ZomboidRLBridge] No active player found; writing fallback telemetry payload.")
            lastMissingPlayerLogTime = now
        end
    elseif payload.invalidPlayerValues and (now - lastInvalidPayloadLogTime) >= 10.0 then
        print("[ZomboidRLBridge] Player values were unavailable; writing fallback payload.")
        lastInvalidPayloadLogTime = now
    end

    writeTelemetryFile(payload)
    print(string.format(
        "[ZomboidRLBridge] x=%.3f y=%.3f z=%.3f health=%.3f missingPlayer=%s",
        payload.x, payload.y, payload.z, payload.health, tostring(payload.missingPlayer)
    ))
end

local function onClientTick()
    local currentTime = nowSeconds()
    if lastLogTime == 0.0 then
        lastLogTime = currentTime
    end
    if (currentTime - lastLogTime) >= LOG_INTERVAL_SECONDS then
        logTelemetry()
        lastLogTime = currentTime
    end
end

Events.OnTick.Add(onClientTick)
print("[ZomboidRLBridge] Telemetry initialized")

