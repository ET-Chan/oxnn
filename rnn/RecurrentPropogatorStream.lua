require ('oxnn')
require ('graph')
require ('misc/EventPool')

local RecurrentPropagator, parent = torch.class('oxnn.RecurrentPropagatorStream', 'oxnn.RecurrentPropagator')


local NOT_SET = parent.NOT_SET

---------------------------------------------------------
--helper function definition
--
local function ref_find(ref)
   return table.pack(ref:find('([iom])([0-9]+):*([0-9]*)'))
end
--TODO: remove this.
local function location_parse(ref)
   if (torch.type(ref) == 'oxnn.RecurrentPropagator_StackName'
         and ((ref.name and ref.name:sub(1,1) == 'i')
              or (ref.name2 and ref.name2[1] == 'i')))
      or torch.type(ref) == 'oxnn.RecurrentPropagator_StackElement' then

      ref = ref.name2 or ref.name
   end
   if torch.type(ref) == 'table' then
      return ref
   end

   assert(torch.type(ref) == 'string', 'Bad stack reference.')
   local _, _, what, i, j = unpack(ref_find(ref))
   assert(what)
   i = tonumber(i) or 1
   j = tonumber(j) or 1
   return {what, i, j}
end


local function outsCheck1(outs)
   -- outs need to be reversible and we can store only at the top of a stack,
   -- therefore is there are N stores to the same stack, with N>1, they need to
   -- be numbered from N to 1 precisely in this order
   outs = torch.type(outs) == 'table' and outs or { outs }
   outs = oxnn.flatten(outs)

   local stacks = {}
   for _,v in ipairs(outs) do
      local what, i, j = unpack(location_parse(v))
      stacks[what..i] = stacks[what] or {}
      table.insert(stacks[what..i], j)
   end
   for k,v in pairs(stacks) do
      for i=1,#v do
         assert(v[#v-i+1] == i, 'Bad outs. See comment above. ' ..
                                '('..k..':'..i..')')
      end
   end
   return true
end


local function RecursiveTensorAdd(dest, src)
   oxnn.RecursiveTensorAdd(dest, src, NOT_SET)
end

local function is_not_set(v)
   if torch.type(v) == 'table' then
      for _,e in pairs(v) do if is_not_set(e) then return true end end
   end
   return v == NOT_SET
end

function table.flatten(arr)
  local result = { }
  
  local function flatten(arr)
    for _, v in ipairs(arr) do
      if torch.type(v) == "table" then
        flatten(v)
      else
        table.insert(result, v)
      end
    end
  end
  
  flatten(arr)
  return result
end

local function print_edge(e)
   local function delendl(s)  -- remove last endl
      if s[#s] == '\n' then s = s:sub(1,#s-1) end
      return s
   end
   local function format_io(io)
      if torch.type(io) == 'table' then
         local str = {}
         for i,v in ipairs(io) do
            if torch.type(v) == 'table' then
               table.insert(str, format_io(v))
            else
               table.insert(str, delendl(tostring(v)))
            end
         end
         return '{'..table.concat(str, ',')..'}'
      else
         return delendl(tostring(io))
      end
   end

   local c = sys.COLORS
   print(c.green..'EDGE:', format_io(e[1]), delendl(tostring(e[2])), format_io(e[3]), c.none)
end

------------------------------------------------------------------


--- Enable CUDA stream function.
-- @param streams, the available streams rp can use.
-- 
-- 
function RecurrentPropagator:__init(cg)
  parent.__init(self, cg)
  self._event_pool = EventPool()
  self.tolerateExcessive = true
end


function RecurrentPropagator:enableStream(streams)
  self.streams = streams
  if (self._dummy_event == nil) then
    local de = {}
    local ev = CudaEvent()
    local mt = {__index = function () return ev end}
    setmetatable(de, mt)
    self._dummy_event = de
  end
end

function RecurrentPropagator:disableStream()
  self.streams = nil
end


---overloading _s_store
--due to the fact that we need to overload the behaviour while dealing
--with CudaEvent
function RecurrentPropagator:_s_store(ref, froms, data, add, addhelper)
   if torch.type(ref) == 'table' then
      if data ~= NOT_SET and torch.type(data) ~= 'CudaEvent' then
         local shrink_data = {}
         if (#ref ~= #data and self.tolerateExcessive) then 
            for i = 1, #ref do
              shrink_data[i] = data[i]
            end
            data = shrink_data
         end
         assert(#ref == #data, 'Number of store elements mismatch.')
         for i, r in ipairs(ref) do
            self:_s_store(r, froms, data[i], add, addhelper)
         end
      else
         -- data == NOT_SET
         -- we are populating the structure with the same data value
         for i, r in ipairs(ref) do
            self:_s_store(r, froms, data, add, addhelper)
         end
      end
      return
   end

   local what, i, j = unpack(location_parse(ref))
   if self.debug > 3 then print(froms, what) end
   local from = froms[what]
   from[i] = from[i] or {}
   if not add then
      j = 1
      --assert(j == 1, 'can only store at the top of the stack')
      table.insert(from[i], data)
   else
      addhelper[what] = addhelper[what] or {}
      if what == 'i' and (from[i] == NOT_SET
            or torch.type(from[i]) ~= 'table') then
         if from[i] == NOT_SET then
            from[i] = data
            addhelper[what][i] = false
         else
            if not addhelper[what][i] then
               from[i] = from[i]:clone()
               addhelper[what][i] = true
            end
            from[i]:add(data)
         end
      else
         local fro = from[i]
         addhelper[what][i] = addhelper[what][i] or {}
         if fro[ #(fro) - j + 1 ] == NOT_SET then
            fro[ #(fro) - j + 1 ] = data
            addhelper[what][i][ #(fro) - j + 1 ] = false  -- was not cloned yet
         else
            if type(fro[ #(fro) - j + 1 ]) ~= 'number'
                  and not addhelper[what][i][ #(fro) - j + 1 ] then
               fro[ #(fro) - j + 1 ] = oxnn.recursiveClone(fro[ #(fro) - j + 1 ])
               addhelper[what][i][ #(fro) - j + 1 ] = true
            end
            RecursiveTensorAdd(fro[ #(fro) - j + 1 ], data)
            --fro[ #(fro) - j + 1 ]:add(data)
         end
      end
   end
   if self.debug > 1 then
      print(add and 'storeADD' or 'store', ref)
   end
   if self.debug > 3 then print(froms) end
end


--TODO: there are many redundancies between the original code
--refactor them.
function RecurrentPropagator:updateOutput(input)
  if (self.streams == nil) then -- stream function is disabled
    return parent.updateOutput(self, input)
  end
  
  local _p_stream = cutorch.getStream()
  --stream is activated
  if self.debug > 0 then print ('updateOutput(stream)') end
  --------------------------------------------------------
  --Cleaning states
  self._GradOutputs = nil
  self._Inputs = nil
  self._inputs = nil
  self._outputs = nil
  self._last_output = nil
  self.output = nil
  self.gradInput = nil
  self._modules_run_count = nil
  --------------------------------------------------------
  --Init states
  self._Inputs = {}
  self._inputs = input
  self._outputs = {}
  self._events = {}
  self._event_pool:freeAllEvent()
  self._modules_run_count = {}
  self._module_clones = self._module_clones or {}
  local mcs = self._module_clones
  local modrc = self._modules_run_count
  local cg = torch.type(self._cg) == 'function'
                and self._cg(input, self._type) or self._cg
  
  self._cg_last = cg
  self._par_cgs_last = {}
  ---------------------------------------------------------
  -- clone modules
  do
     local rc = {}   -- projected run count, ie clones needed
     for idx = 1, #cg do
        local e = cg[idx]
        local _, mod_, _ = unpack(e)
        rc[mod_] = rc[mod_] and rc[mod_] + 1 or 1
     end

     local clones_needed = 0
     for k,v in pairs(rc) do
        if type(k) == 'string' then
           mcs[k] = mcs[k] or {}
           local new_clone_count = rc[k] - #(mcs[k])
           clones_needed = clones_needed + math.max(0, new_clone_count)
        end
     end

     local clones_made = 0
     if clones_needed > 0 then
        --print('Cloning '..clones_made..'/'..clones_needed)
     end
     for k,v in pairs(rc) do
        if type(k) == 'string' then
           mcs[k] = mcs[k] or {}
           local new_clone_count = rc[k] - #(mcs[k])
           if new_clone_count > 0 then
              --print(k)
              local orig = self:_s_get(k, {m=self.modules})
              if not orig.__oxnn_RP_createCloneAndShareAll then
                 local new_clones =
                    RecurrentPropagator.cloneManyTimes(orig, new_clone_count)
                 for _,clone in ipairs(new_clones) do
                    table.insert(mcs[k], clone)
                 end
              else
                 for i=1,new_clone_count do
                    table.insert(mcs[k], orig:__oxnn_RP_createCloneAndShareAll())
                    --collectgarbage()
                 end
              end
              collectgarbage()
              clones_made = clones_made + new_clone_count
              if clones_needed > 0 then
--                 print('Cloning '..clones_made..'/'..clones_needed)
              end
           end
        end
     end
     --if printing then print 'clones created' end
      ----------------------------------------------------------------
     --Streamified forward propagation
     --Only used flatten storage.
     local froms = {i=self._inputs, f=self._outputs}
     local froms_store = {f=self._outputs}
     local froms_events = {f=self._events, i=self._dummy_event}
     local streams = self.streams
     --first, output the computation graph, and flatten all stacks
     local g, _s_map, _elem_sz = self:getDependencyGraph(false, true)
     --store it for updateGradInput
     self._g_last = g
     --scan through the whole graph, finding those without any ancesters
     local par_cg = {}

     for _, node in ipairs(g.nodes) do
      if (node.ansc_ == nil or node.ansc_ == 0) then
        par_cg[#par_cg + 1] = node
      end
     end
--     print (#g.nodes)
     --now, do the parallel thing
     local _cal_cgs = 0
--     print (#cg)
--     os.exit(-1)
     local cur_stream = 0
     while (#cg ~= _cal_cgs) do
--      print (#cg, _cal_cgs)
      assert(#par_cg~=0)
      --push the par_cg into last_par_cgs, for backward calculation.
      table.insert(self._par_cgs_last, par_cg)
      for i, v in ipairs(par_cg) do
        --it is now almost the same with normal rp:forward
        local e = v.data
        local idx = v.idx
        local ins_, mod_, outs_ = unpack(e)
        if self.debug > 2 then
          io.write(sys.COLORS.red)
          print(table.concat(u.rep('-', 80), ''))
          io.write(sys.COLORS.none)
        end
        if self.debug > 0 then
            print_edge(e)
        end
        if not self.fast then
          assert(outsCheck1(outs_))
        end
        local ins = self:_s_get(ins_, froms)
        modrc[mod_] = modrc[mod_] and modrc[mod_] + 1 or 1
        local mod = mcs[mod_] and mcs[mod_][modrc[mod_]] or mod_
        self._Inputs[idx] = ins
        if self.debug > 2 then
           io.write(sys.COLORS.yellow)
           oxnn.recprint(ins, 'ins')
           io.write(sys.COLORS.none)
           oxnn.recprint(mod)
        end
        ---------------
        cur_stream = cur_stream + 1
        cutorch.setStream(streams[cur_stream % #streams + 1])
        --before forwarding, get all events need to wait
        local evs_ = self:_s_get(ins_, froms_events)
        evs_ = torch.type(evs_) == 'table' and evs_ or {evs_}
        evs_ = table.flatten(evs_)
        for _, v in ipairs(evs_) do
          assert(torch.type(v) == 'CudaEvent')
          v:streamWaitOn()
        end
        self.output = mod:forward(ins)
        ----sync disabled
        if self.debug > 2 then
           io.write(sys.COLORS.cyan)
           oxnn.recprint(self.output, 'output')
           io.write(sys.COLORS.none)
        end
        self:_s_store(outs_, froms_store, self.output)
        --get a free event from pool
        local ev_ = self._event_pool:getNewEvent()
        --register on current stream
        ev_:record()
        --store it on events stack
        self:_s_store(outs_, froms_events, ev_)
        if (idx == #cg) then
          self._last_output = self.output
        end
      end
      ---bubble section, add as much CPU
      --computation here as possible
      
      _cal_cgs = _cal_cgs + #par_cg
      local next_par_cg = {}
      for i, v in ipairs(par_cg) do
        for _,child in ipairs(v.children) do
          child.ansc_ = child.ansc_ - 1
          if (child.ansc_ == 0) then
            table.insert(next_par_cg, child)
          end
        end
      end
      
      par_cg = next_par_cg
     end
  end  
  
  if self.debug > 0 then print ('updateOutput end') end
  self.output = self._last_output
  cutorch.setStream(_p_stream)
  return self.output
  
end

function RecurrentPropagator:updateGradInput(input, gradOutput)
  if not self.streams then
    return parent.updateGradInput(self, input, gradOutput)
  end
  if self.debug > 0 then print('updateGradInput (stream)') end
  if not self.fast then
    assert(input == self._inputs)
  end
  local cg = self._cg_last
  local _p_stream = cutorch.getStream()
  self._GradOutputs = {}
  self._event_pool:freeAllEvent()
  local function recursiveCloneStacks(t, depth)
      depth = depth or 0
      local clone
      if depth < 2 and torch.type(t) == 'table' then
         clone = {}
         for i = 1, #t do
            clone[i] = recursiveCloneStacks(t[i], depth+1)
         end
      else
         if torch.typename(t) and torch.typename(t):find('torch%..+Tensor') then
            clone = NOT_SET
         elseif depth >= 1 then
            clone = NOT_SET
         else
            error('Unimplemented feature.')
         end
      end
      return clone
   end
   
   local gradInputs_i = recursiveCloneStacks(self._inputs)
   local gradInputs_f = {}
   local events = {}
   local events_i = recursiveCloneStacks(self._inputs)
   local streams = self.streams
   local cur_stream = 0
   local froms_events = {f = events, i=events_i}
   
   if self.debug > 0 then print 'simulate outputs' end
   
   for idx = 1, #cg do
      local e = cg[idx]
      local ins_, mod_, outs_ = unpack(e)
      if idx < #cg then
         self:_s_store(outs_, {f=gradInputs_f}, NOT_SET)
      else
         self:_s_store(outs_, {f=gradInputs_f}, gradOutput)
      end
      
      --allocate `different' event handlers for outs_
      outs_ = torch.type(outs_) == 'table' and outs_ or { outs_ }
      outs_ = table.flatten(outs_)
      for _, v in ipairs(outs_) do
        self:_s_store(v, {f=events}, self._event_pool:getNewEvent())
      end
   end
   --then , allocate `different' event handlers for all inputs_
   local function recursiveAssign(t, func)
      if t ~= NOT_SET then
        for i = 1, #t do
          t[i] =  recursiveAssign(t[i], func)
        end
        return t
      else
        assert(t == NOT_SET)
        return func()
      end
   end
   
   recursiveAssign(events_i,
      function () return self._event_pool:getNewEvent() end)
   if self.debug > 0 then print'grad input calculation' end
   local modrc = {}   -- local only
   local mcs = self._module_clones
   local froms_unstore = {f=gradInputs_f}
   local froms_store = {i=gradInputs_i, f=gradInputs_f}
   local store_addhelper = {}
   local cg_batch = self._par_cgs_last
   local _idx_map = {}
   self._idx_map_last = _idx_map
   for k = #cg_batch , 1, -1 do
    local par_cg = cg_batch[k]
    --this needs to be reversed as well
    for i = #par_cg, 1, -1 do
      local v = par_cg[i]
      local e = v.data
      local idx = v.idx
      _idx_map[e] = idx
      local ins_, mod_, outs_ = unpack(e)
      if self.debug > 2 then
         io.write(sys.COLORS.red)
         print(table.concat(u.rep('-', 80), ''))
         io.write(sys.COLORS.none)
      end
      if self.debug > 0 then print_edge(e) end
      
      local ins = self._Inputs[idx]
      modrc[mod_] = modrc[mod_] and modrc[mod_] + 1 or 1
      local mod = mcs[mod_] and
                     mcs[mod_][self._modules_run_count[mod_] - modrc[mod_]+1]
                     or mod_
      if not self.fast then assert(mod) end
      ------------------------------------
      cur_stream = cur_stream + 1
      cutorch.setStream(streams[cur_stream % #streams + 1])
      --before backwarding, get all events need to wait
      local evs_ = self:_s_get(outs_, froms_events)
      evs_ = torch.type(evs_) == 'table' and evs_ or {evs_}
      evs_ = table.flatten(evs_)
      for _, v in ipairs(evs_) do
          assert(torch.type(v) == 'CudaEvent')
          v:streamWaitOn()
      end
      --then, we shall go backwarding now.
      local gradOutput = self:_s_get(outs_, {i=gradInputs_i, f=gradInputs_f})
      if not self.fast then
         assert(not is_not_set(gradOutput)
                or torch.isTypeOf(mod, oxnn.CriterionTable),
                'Missing gradOutput. Expecting already computed gradOutput ' ..
                'or this module to be oxnn.CriterionTable')
      end
      self._GradOutputs[idx] = gradOutput
      if self.debug > 2 then
         io.write(sys.COLORS.yellow)
         oxnn.recprint(ins, 'ins')
         io.write(sys.COLORS.magenta)
         oxnn.recprint(gradOutput, 'gradOutput')
         io.write(sys.COLORS.none)
      end
      local gradInput = mod:updateGradInput(ins, gradOutput)
      if self.debug > 2 then
         io.write(sys.COLORS.cyan)
         oxnn.recprint(gradInput, 'gradInput')
         io.write(sys.COLORS.none)
      end
      
      self:_s_unstore(outs_, froms_unstore)
      --before storing, check all ins_ are already finished
      evs_ = self:_s_get(ins_, froms_events)
      evs_ = torch.type(evs_) == 'table' and evs_ or {evs_}
      evs_ = table.flatten(evs_)
      for _, v in ipairs(evs_) do
          assert(torch.type(v) == 'CudaEvent')
          v:streamWaitOn()
      end
      self:_s_store(ins_, froms_store, gradInput, true, store_addhelper)
      for _, v in ipairs(evs_) do
        v:record()
      end
    end
--    print (#par_cg)
--    cutorch.synchronize()
   end
   
   local function zero(i, gi, idx)
      if gi[idx] == NOT_SET then
         gi[idx] = oxnn.recursiveClone(i[idx],
                                      function (t)
                                          return type(t)=='number' and 0
                                             or t.new():resizeAs(t):zero() end)
      elseif torch.type(i[idx]) == 'table' then
         for k,_ in pairs(i[idx]) do
            zero(i[idx], gi[idx], k)
         end
      elseif torch.type(i[idx]) == 'number' then
         gi[idx] = 0
      elseif torch.type(i[idx]):match('torch%..*Tensor') then
      else
         error('This shouldn\'t happen')
      end
   end
   
   --synchronize with all gradInputs_i ops, otherwise, error is possible
--   cutorch.synchronize()
   cutorch.setStream(_p_stream)
   for i,_ in ipairs(gradInputs_i) do
      zero(self._inputs, gradInputs_i, i)
   end
   cutorch.synchronize()--the final one might be not finished.
   self.gradInput = gradInputs_i
   if self.debug > 0 then print('updateGradInput end') end
   return self.gradInput
end


---Return dependency graphs according to self._cg_last and 
-- self._module_clones
-- That means, in other words, you have to at least 
-- forward the model once (or prepare all the clones and cgs)
-- to generate the dependency graph. Also, this module does not
-- check if stacks are accessed properly
-- @param, boolean, pretty, states whether needs to prettify
-- the nodes' label, for graphviz use, default is false
-- @return the graph, compatiable with torch.graph
function RecurrentPropagator:getDependencyGraph(pretty, flatten)
  pretty = pretty or false
  local cg = self._cg_last
  local outs_to_cg = {}
  local cg_to_nodes = {}
  local g = graph.Graph()
  local mcs = self._module_clones
  local stacks = {}
  local counter = 0
  local flatten_storage = {}
--  local cg_to_outs = {}
  for idx = 1, #cg do
    local e = cg[idx]
    local ins_, mod_, outs_ = unpack(e)
    outs_ = torch.type(outs_) == 'table' and outs_ or { outs_ }
    ins_ = torch.type(ins_) == 'table' and ins_ or {ins_}
    outs_ = table.flatten(outs_)
    ins_  = table.flatten(ins_)
--    print ('edge: ', idx, #outs_, #ins_)
    local innode = graph.Node(e)
    table.insert(g.nodes, innode)
    g.nodes[innode] = #g.nodes
    innode.idx = idx
    cg_to_nodes[e] = innode
    if (pretty) then
      innode.label = function ()
        return tostring(mcs[mod_] and mcs[mod_][1] or mod_)
      end
    end
    
    for _, v in ipairs(ins_) do
      local what, i, j = unpack(location_parse(v))
      local stack = stacks[what..i]
  --    print ('get: ' .. table.concat({what, i,j},':'))
      if (what ~= 'i') then
        local outnode = cg_to_nodes[flatten_storage[stack[#stack - j + 1]]]
        if (not outnode.children[innode]) then--prevent duplicated edges.
          g:add(graph.Edge(outnode, innode))
          --this is used for topology sort.
          innode.ansc_ = innode.ansc_ and innode.ansc_ + 1 or 1 
        end
        if (flatten) then
          v.name2 = {'f', stack[#stack - j + 1], 1}
        end
      end
    end
    for _, v in ipairs(outs_) do
     local what, i, j = unpack(location_parse(v))
     stacks[what..i] = stacks[what..i] or {}
     counter = counter + 1
     table.insert(stacks[what..i], counter)
     flatten_storage[counter] = e
     if (flatten) then
      v.name2 = {'f', counter, 1}
     end
    end
  end
  return g, stacks, counter
end


function RecurrentPropagator:_accGradParameters(input, gradOutput, scaleOrLr,
                                                tocall)
   if (not self.streams) then
    parent._accGradParameters(self, input, gradOutput, scaleOrLr, tocall)
    return
   end
   
   local _p_stream = cutorch.getStream()
   if self.debug > 0 then print('_acc(Update)GradInput (stream)') end
   if not self.fast then
      assert(input == self._inputs) -- perhaps too strict
   end
   local streams = self.streams
   local cg = self._cg_last
   if self.debug > 0 then
      print'acc (update) grad parameters calculation (stream)'
   end
   local modrc = {}   -- local only
   --assign to different streams based on the amount of weights.
   local cg_batch = {}
   local assigned_streams = {}
   local counter = 1
   for idx = 1, #cg do
    --group by mod
    local e = cg[idx]
    local _, mod_,_ = unpack(e)
    if (assigned_streams[mod_] == nil) then
      assigned_streams[mod_] = counter
      counter = counter + 1
      cg_batch[mod_] = {}
    end
    table.insert(cg_batch[mod_], e)
   end
   
   counter = 0
   local mcs = self._module_clones
   local g = self._g_last
   local _idx_map = self._idx_map_last
   while (counter ~= #cg) do
    assert(next(cg_batch) ~= nil)
    local _delete = {}
    for _, par_cg in pairs(cg_batch) do
      assert(#par_cg ~= 0)
      --take the last element from par_cg
      local e = par_cg[#par_cg]
      local idx = _idx_map[e]
      local ins_, mod_, outs_ = unpack(e)
      par_cg[#par_cg] = nil
      if (#par_cg == 0) then table.insert(_delete, mod_) end
      counter = counter + 1
      --set to the corresponding stream
      cutorch.setStream(streams[assigned_streams[mod_] % #streams + 1])
      --do cool thing.
      local ins = self._Inputs[idx]
      modrc[mod_] = modrc[mod_] and modrc[mod_] + 1 or 1
      local mod = mcs[mod_] and
                      mcs[mod_][self._modules_run_count[mod_] - modrc[mod_]+1]
                      or mod_
      if not self.fast then assert(mod) end    
      
      local gradOutput = self._GradOutputs[idx]
      mod[tocall](mod, ins, gradOutput, scaleOrLr)
    end  
    for _, v in ipairs(_delete) do
      cg_batch[v] = nil
    end
   end
   cutorch.setStream(_p_stream)
   if self.debug > 0 then print('_acc(Update)GradInput end') end
end



--flatten all the stacks on a continuous table.
--
--function RecurrentPropagator:flattenStacks(stacks, counter) 
--  local cg = self._cg_last
--  for idx = 1, #cg do
--    local e = cg[idx]
--    local ins_, mod_, outs_ = unpack(e)
--    outs_ = torch.type(outs_) == 'table' and outs_ or { outs_ }
--    ins_ = torch.type(ins_) == 'table' and ins_ or {ins_}
--    outs_ = table.flatten(outs_)
--    ins_  = table.flatten(ins_)
--    
--    for _, v in ipairs(ins_) do
--      local what, i, j = unpack(location_parse(v))
--      local stack = stacks[what..i]
--  --    print ('get: ' .. table.concat({what, i,j},':'))
--      if (what ~= 'i') then
--        v.name2 = {'f',(counter -  stacks[what..i][j] + 1),1}
--      end
--    end
--    
--    for _, v in ipairs (outs_) do
--      local what, i, j = unpack(location_parse(v))
--      v.name2 = {'f' , (counter - stacks[what..i][j] + 1),1}
--    end
--    
--  end
--end

function RecurrentPropagator:GetIO(ref)
  local froms
  if (self.streams) then
    froms = {i = self._inputs,  f=self._outputs}
  else
    froms = {i = self._inputs,  o=self._outputs}
  end
  return self:_s_get(ref, froms)
end
