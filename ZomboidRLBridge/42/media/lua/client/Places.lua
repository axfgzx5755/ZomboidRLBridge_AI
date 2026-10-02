-- Semantic labels for loaded tiles only; no claim that the entire building is seen.
ZomboidRLPlaces = {}
local function quote(value)
    return '"'..tostring(value):gsub("\\","\\\\"):gsub('"','\\"'):gsub("[%c]"," ")..'"'
end
function ZomboidRLPlaces.scan(x,y,z,radius)
    local entries={}
    for sy=y-radius,y+radius do
        for sx=x-radius,x+radius do
            local square=getCell():getGridSquare(sx,sy,z)
            if square then
                local room=square:getRoom()
                local building=square:getBuilding()
                local bid,rid,name="","",""
                if building then
                    local def=building:getDef()
                    bid=string.format("%d:%d",def:getX(),def:getY())
                end
                if room then
                    local def=room:getRoomDef()
                    rid=bid..":"..tostring(z)..":"..string.format("%.0f",def:getID())
                    name=def:getName() or ""
                end
                entries[#entries+1]=string.format('{"x":%d,"y":%d,"building":%s,"room":%s,"name":%s}',
                    sx,sy,quote(bid),quote(rid),quote(name))
            end
        end
    end
    return '{"schema":1,"tiles":['..table.concat(entries,",")..']}'
end
