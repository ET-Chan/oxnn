require ('misc/CudaEvent')

local EventPool, _ = torch.class('EventPool')

function EventPool:__init()
  self._pools_aval = {}
  self._pools_inuse = {}
end

function EventPool:getNewEvent()
  local ret = next(self._pools_aval)
  if (ret == nil) then
    ret = CudaEvent()
  else
    self._pools_aval[ret] = nil
  end
  
  self._pools_inuse[ret] = true
  return ret
end

function EventPool:freeEvent(ce)
  assert(self._pools_aval[ce] == nil)
  assert(self._pools_inuse[ce] ~= nil)
  self._pools_aval[ce] = true
  self._pools_inuse[ce] = nil
end

function EventPool:freeAllEvent()
  for k, v in pairs(self._pools_inuse) do
    assert(self._pools_aval[k] == nil)
    self._pools_aval[k] = true
  end
  self._pools_inuse = {}
end