require 'lstmlm.util'
lstm = require 'lstmlm.LSTM'
require 'nn'

local model = lstm.lstm(10, 20, 2, 0.1)
