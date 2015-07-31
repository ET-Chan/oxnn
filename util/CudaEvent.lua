require 'oxnn'


--This is a wrapper for cudaevent
--It is handy, as the __gc is rewritten
--That is, while event is collected
--destroyEvent is automatically called
--It is user resposinbility to init cuda
--before requiring this library

local CudaEvent, _ = torch.class('CudaEvent') 


local finalizer =  function (ev)
  oxnn.oxnn_event_destroyEvent(ev)
--  print ('should not happened')
end

function CudaEvent:__init()
  assert(oxnn.oxnn_event_createEvent, 'oxnn.InitCuda() has not been called yet.')
  self.ev = oxnn.oxnn_event_createEvent()
  getmetatable(self.ev).__gc = finalizer
  
  --setmetatable(self.ev, _mt_for_evs)
end

--keep a snapshot of current stream into the event
function CudaEvent:record()
  oxnn.oxnn_event_recordEvent(self.ev)
end

--fire an operation into current stream, to wait on this event
function CudaEvent:streamWaitOn()
  oxnn.oxnn_event_streamWaitEvent(self.ev)
end


