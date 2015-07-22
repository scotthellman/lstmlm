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
