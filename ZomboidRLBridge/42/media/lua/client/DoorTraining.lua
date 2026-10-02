-- Single ordinary door training. Called only through the leased training bridge.
ZomboidRLDoor = {}
local function adjacent(player, door)
    local s = door:getSquare()
    local x, y = math.floor(player:getX()), math.floor(player:getY())
    local ox = s:getX() - (door:getNorth() and 0 or 1)
    local oy = s:getY() - (door:getNorth() and 1 or 0)
    return math.floor(player:getZ()) == s:getZ() and
        ((x == s:getX() and y == s:getY()) or (x == ox and y == oy))
end
local function suitable(door)
    local sprite = door:getSprite()
    local props = sprite and sprite:getProperties()
    if door:isDestroyed() then return false, "destroyed door" end
    if door:isBarricaded() then return false, "barricaded door" end
    if door:isLocked() then return false, "locked door" end
    if door:isLockedByKey() then return false, "locked-by-key door" end
    -- Build 42 PropertyContainer uses has(), as in the installed game Lua.
    if props and props:has("DoubleDoor") then return false, "double door" end
    if props and props:has("GarageDoor") then return false, "garage door" end
    return true
end
function ZomboidRLDoor.handle(player, operation, x, y, z, north, id)
    local matches = {}
    local nearby = {}
    local scan = operation == "door_scan"
    local r = scan and 2 or 0
    for sy = math.floor(y)-r, math.floor(y)+r do
        for sx = math.floor(x)-r, math.floor(x)+r do
            local square = getCell():getGridSquare(sx, sy, z)
            if square then
                local objects = square:getObjects()
                for i = 0, objects:size()-1 do
                    local obj = objects:get(i)
                    if scan and instanceof(obj, "IsoDoor") then
                        local s = obj:getSquare()
                        local ox = s:getX() - (obj:getNorth() and 0 or 1)
                        local oy = s:getY() - (obj:getNorth() and 1 or 0)
                        nearby[#nearby+1] = string.format("door=(%d,%d); stand on (%d,%d) or (%d,%d)",
                            s:getX(),s:getY(),s:getX(),s:getY(),ox,oy)
                    end
                    if instanceof(obj, "IsoDoor") and
                        ((scan and adjacent(player,obj)) or
                        (not scan and obj:getNorth() == (north == 1))) then
                        matches[#matches+1] = obj
                    end
                end
            end
        end
    end
    if #matches ~= 1 then
        local details = ""
        if scan then
            details = string.format("; player tile=(%d,%d,%d); ",
                math.floor(player:getX()),math.floor(player:getY()),math.floor(player:getZ()))
            details = details .. (#nearby > 0 and table.concat(nearby," | ") or
                "no IsoDoor within 2 tiles; move closer to a normal building door (crafted/modded doors may be unsupported)")
        end
        error("expected exactly one adjacent ordinary door; found " .. #matches .. details)
    end
    local door = matches[1]
    local allowed, reason = suitable(door)
    if not allowed then error("unsupported training door: " .. reason .. "; use an unlocked, unbarricaded, intact single door") end
    if operation == "door_close" then
        if not adjacent(player,door) then error("reset position must be beside the door") end
        if door:IsOpen() then door:ToggleDoor(player) end
        if door:IsOpen() then error("door did not close; check obstruction") end
    elseif operation == "door_open" and adjacent(player,door) and not door:IsOpen() then
        door:ToggleDoor(player)
    end
    local s = door:getSquare()
    local file = getFileWriter("pz_door_state.txt", true, false)
    if not file then error("cannot write door state") end
    local ok, err = pcall(function()
        file:write(string.format('{"id":"%s","schema":1,"x":%d,"y":%d,"z":%d,"north":%s,"open":%s,"locked":%s,"adjacent":%s}',
            id,s:getX(),s:getY(),s:getZ(),tostring(door:getNorth()),tostring(door:IsOpen()),
            tostring(door:isLocked() or door:isLockedByKey()),tostring(adjacent(player,door))))
    end)
    file:close()
    if not ok then error(err) end
end
