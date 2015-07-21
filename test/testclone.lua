require 'lstmlm.util'
require 'nn'

local model = nn.Sequential()
model:add(nn.Linear(10, 10))
model:add(nn.Tanh())

local diskClones = cloneThroughTimeDisk(model, 4)
local clones = cloneThroughTime(model, 4)

local newVal = 10
model:parameters()[1][1][1] = newVal

assert(diskClones[1]:parameters()[1][1][1] == newVal)
assert(clones[1]:parameters()[1][1][1] == newVal)
