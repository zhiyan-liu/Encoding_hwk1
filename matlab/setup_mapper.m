%% Config all the parameters.
ch_conf.b = 1;
ch_conf.rho = 1;
mapping_conf.M = [1, 1j, -1j, -1];  % grey code with Ps=1.
mapping_conf.bps = log2(length(mapping_conf.M));
mapping_conf.mode = 'ch_known'; % ['ch_unknown', 'ch_known']
mapping_conf.alg = 'zf';        % ['zf', 'mse'];
mapping_conf.out = 'L2';        % ['hard', 'LLR', 'L2']
mapping_conf.pilotrate = 4;     % one pilot every N symbols
