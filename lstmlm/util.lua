require 'nn'
require 'nngraph'
local OneHot, parent = torch.class('OneHot', 'nn.Module')

function OneHot:__init(outputSize)
  parent.__init(self)
  self.outputSize = outputSize
  -- We'll construct one-hot encodings by using the index method to
  -- reshuffle the rows of an identity matrix. To avoid recreating
  -- it every iteration we'll cache it.
  self._eye = torch.eye(outputSize)
end

function OneHot:updateOutput(input)
  self.output:resize(input:size(1), self.outputSize):zero()
  if self._eye == nil then self._eye = torch.eye(self.outputSize) end
  self._eye = self._eye:float()
  local longInput = input:long()
  self.output:copy(self._eye:index(1, longInput))
  return self.output
end

function cloneThroughTime(net, n)
    --[[for backprop through n timesteps, we need to replicate our network n
        times, sharing the weights across networks
    --]]
    local clones = {}
    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    --for each timestep, clone the network
    for t = 1, n do
        clones[t] = net:clone()
        if net.parameters then
            --share the parameters across the clones
            local cloneParams, cloneGradParams = clones[t]:parameters()
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
        end
    end

    return clones
end
