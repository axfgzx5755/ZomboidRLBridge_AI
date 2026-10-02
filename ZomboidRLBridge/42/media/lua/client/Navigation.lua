-- Read-only navigation geometry. Unknown/unloaded cells are blocked.
ZomboidRLNavigation = {}
local directions = {{0,-1},{1,0},{0,1},{-1,0}}
function ZomboidRLNavigation.scan(x, y, z, radius)
    x, y = math.floor(x), math.floor(y)
    local cell = getCell()
    local values = {}
    local function free(square)
        return square and square:TreatAsSolidFloor() and square:isFree(false)
    end
    for sy = y-radius, y+radius do
        for sx = x-radius, x+radius do
            local square = cell:getGridSquare(sx, sy, z)
            local passable = free(square)
            values[#values+1] = passable and "0" or "1"
            for _, d in ipairs(directions) do
                local other = cell:getGridSquare(sx+d[1], sy+d[2], z)
                local blocked = not passable or not free(other)
                if not blocked then
                    -- Check both sides: doors, windows and walls are edges, not occupied tiles.
                    blocked = square:isBlockedTo(other) or other:isBlockedTo(square)
                        or square:isWindowTo(other) or other:isWindowTo(square)
                end
                values[#values+1] = blocked and "1" or "0"
            end
        end
    end
    return string.format('{"schema":2,"x":%d,"y":%d,"z":%d,"radius":%d,"cells":[%s]}',
        x,y,z,radius,table.concat(values,","))
end
