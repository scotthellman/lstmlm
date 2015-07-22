--[[ 
Long Short-Term Memory unit, shamelessly stolen from https://github.com/karpathy/char-rnn
--]]
local LSTM = {}
function LSTM.lstm(input_size, rnn_size, num_layers, dropout)
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  -- 2n because each layer will take in its internal state and output from the previous timestep.
  -- +1 because of the actual input to the whole network
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- the initial input to the network
  for L = 1,num_layers do
    table.insert(inputs, nn.Identity()()) -- prev_state[L]
    table.insert(inputs, nn.Identity()()) -- prev_output[L]
  end

  local input, input_size_L
  local outputs = {}
  for L = 1,num_layers do
    -- get the output and state of this layer from the previous timestep
    local prev_output = inputs[L*2+1]
    local prev_state = inputs[L*2]

    -- grab the input to this layer
    -- if we're in the first hidden layer, then use the input to the whole network
    -- otherwise, use the output of the previous hidden layer
    if L == 1 then 
      input = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else 
      input = outputs[(L-1)*2] --even indices in outputs are layer outputs, odd indices are internal state
      if dropout > 0 then input = nn.Dropout(dropout)(input) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- evaluate the input sums at once for efficiency.
    -- all of the gate inputs take the general form (ignoring subscripts) of: wx + wh
    -- where x is the current layer's input and h is the layer's previous output.
    -- so here, the various xs and hs have been concatenated, allowing us to compute
    -- everything in one go
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(input)
    local out2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_output)
    local all_input_sums = nn.CAddTable()({i2h, out2h})
    -- undo the concatenation to extract the output of the gates
    local sigmoid_chunk = nn.Narrow(2, 1, 3 * rnn_size)(all_input_sums)
    sigmoid_chunk = nn.Sigmoid()(sigmoid_chunk)
    local in_gate = nn.Narrow(2, 1, rnn_size)(sigmoid_chunk)
    local forget_gate = nn.Narrow(2, rnn_size + 1, rnn_size)(sigmoid_chunk)
    local out_gate = nn.Narrow(2, 2 * rnn_size + 1, rnn_size)(sigmoid_chunk)
    -- decode the write inputs
    local in_transform = nn.Narrow(2, 3 * rnn_size + 1, rnn_size)(all_input_sums)
    in_transform = nn.Tanh()(in_transform)

    -- perform the LSTM update
    local next_state           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_state}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_output = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_state)
    table.insert(outputs, next_output)
  end

  -- set up the decoder
  local top_output = outputs[#outputs]
  if dropout > 0 then top_output = nn.Dropout(dropout)(top_output) end
  local proj = nn.Linear(rnn_size, input_size)(top_output)
  local logsoft = nn.LogSoftMax()(proj)
  table.insert(outputs, logsoft)

  return nn.gModule(inputs, outputs)
end

return LSTM
