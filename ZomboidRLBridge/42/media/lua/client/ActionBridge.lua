-- Free-play bridge: observe/open only. Never teleports or changes player stats.
require "Navigation"
require "Places"
local lastPoll, lastId = 0, nil
local function read(path)
    local file = getFileReader(path,false)
    if not file then return nil end
    local value = file:readLine(); file:close(); return value
end
local function words(value)
    local result = {}
    for word in string.gmatch(value or "", "%S+") do result[#result+1]=word end
    return result
end
local function escape(value)
    return tostring(value):gsub("\\", "\\\\"):gsub('"','\\"'):gsub("[%c]"," ")
end
local function flag(value) return value and "true" or "false" end
local function doorStatus(door)
    local sprite = door:getSprite()
    local props = sprite and sprite:getProperties()
    local supported = not door:isDestroyed() and not (props and
        (props:has("DoubleDoor") or props:has("GarageDoor")))
    return supported, door:isLocked() or door:isLockedByKey(), door:isBarricaded()
end
local function snapshot(player)
    local px,py,pz=player:getX(),player:getY(),player:getZ()
    local x,y,z=math.floor(px),math.floor(py),math.floor(pz)
    local doors={}
    for sy=y-4,y+4 do
        for sx=x-4,x+4 do
            local square=getCell():getGridSquare(sx,sy,z)
            if square then
                local objects=square:getObjects()
                for i=0,objects:size()-1 do
                    local obj=objects:get(i)
                    if instanceof(obj,"IsoDoor") then
                        local supported,locked,barricaded=doorStatus(obj)
                        doors[#doors+1]=string.format('{"x":%d,"y":%d,"z":%d,"north":%s,"open":%s,"locked":%s,"barricaded":%s,"supported":%s}',
                            sx,sy,z,flag(obj:getNorth()),flag(obj:IsOpen()),flag(locked),flag(barricaded),flag(supported))
                    end
                end
            end
        end
    end
    local health=player:getHealth()
    if health>1 then health=health/100 end
    local zombies={}
    local zombieList=getCell():getZombieList()
    if zombieList then
        for i=0,zombieList:size()-1 do
            local zombie=zombieList:get(i)
            local zx,zy,zz=zombie:getX(),zombie:getY(),zombie:getZ()
            if math.floor(zz)==z and math.abs(zx-px)<=8 and math.abs(zy-py)<=8 then
                zombies[#zombies+1]=string.format('{"x":%.3f,"y":%.3f}',zx,zy)
            end
        end
    end
    return string.format('"schema":1,"player":{"x":%.6f,"y":%.6f,"z":%.6f,"health":%.6f,"dead":%s,"vehicle":%s},"map":%s,"doors":[%s],"zombies":[%s],"places":%s',
        px,py,pz,health,flag(player:isDead()),flag(player:getVehicle()~=nil),
        ZomboidRLNavigation.scan(x,y,z,4),table.concat(doors,","),table.concat(zombies,","),ZomboidRLPlaces.scan(x,y,z,4))
end
local function openDoor(player,x,y,z,north)
    local px,py=math.floor(player:getX()),math.floor(player:getY())
    local ox,oy=x-(north==1 and 0 or 1),y-(north==1 and 1 or 0)
    if math.floor(player:getZ())~=z or not ((px==x and py==y) or (px==ox and py==oy)) then
        return "not_adjacent"
    end
    local square=getCell():getGridSquare(x,y,z)
    if not square then return "unloaded" end
    local objects=square:getObjects()
    local candidate=nil
    for i=0,objects:size()-1 do
        local obj=objects:get(i)
        if instanceof(obj,"IsoDoor") and obj:getNorth()==(north==1) then
            if candidate then return "ambiguous" end
            candidate=obj
        end
    end
    if not candidate then return "missing" end
    local supported,locked,barricaded=doorStatus(candidate)
    if not supported then return "unsupported" end
    if locked then return "locked" end
    if barricaded then return "barricaded" end
    if candidate:IsOpen() then return "already_open" end
    candidate:ToggleDoor(player)
    return candidate:IsOpen() and "opened" or "blocked"
end
local function tick()
    local now=getTimestampMs()/1000
    if now-lastPoll<0.1 then return end
    lastPoll=now
    local command=words(read("pz_action_command.txt"))
    if #command~=8 or command[1]==lastId then return end
    local id,session,expires,op=command[1],command[2],tonumber(command[3]),command[4]
    local lease=words(read("pz_training_lease.txt"))
    if #lease~=2 or lease[1]~=session or not tonumber(lease[2]) or
        tonumber(lease[2])<=now or tonumber(lease[2])>now+60 or
        not expires or expires<=now or expires>now+30 then return end
    lastId=id
    local ok,result=pcall(function()
        if isClient() or isServer() then error("single-player only") end
        local player=getPlayer()
        if not player then error("no active player") end
        if player:isDead() then error("player dead") end
        if player:getVehicle() then error("leave vehicle") end
        if math.abs(player:getZ())>0.1 then error("ground floor only") end
        local outcome="observed"
        if op=="open" then
            local x,y,z,north=tonumber(command[5]),tonumber(command[6]),tonumber(command[7]),tonumber(command[8])
            for _,v in ipairs({x,y,z,north}) do
                if v~=v or v==math.huge or v==-math.huge or v~=math.floor(v) then error("invalid coordinates") end
            end
            if not x or not y or not z or (north~=0 and north~=1) then error("invalid door") end
            outcome=openDoor(player,x,y,z,north)
        elseif op~="observe" then error("unknown action") end
        return '"outcome":"'..outcome..'",'..snapshot(player)
    end)
    local payload='{"id":"'..escape(id)..'","status":"'..(ok and "ok" or "error")..'",'
    payload=payload..(ok and result or '"error":"'..escape(result)..'"')..'}'
    local file=getFileWriter("pz_action_ack.txt",true,false)
    if not file then error("cannot write action reply") end
    local written,err=pcall(function() file:write(payload) end)
    file:close()
    if not written then error(err) end
end
Events.OnTick.Add(function()
    local ok,err=pcall(tick)
    if not ok then print("[ZomboidRLBridge] Action bridge error: "..tostring(err)) end
end)
print("[ZomboidRLBridge] Free-play action bridge ready")
