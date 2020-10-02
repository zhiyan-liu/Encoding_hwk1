function decoded_bits = conv_decode(encoded_bits, conv_encoder_conf, soft_decode)
% Viterbi decoder for Convolutional Code.
    n = conv_encoder_conf.n;
    k = conv_encoder_conf.k;
    N = conv_encoder_conf.N;
    A = conv_encoder_conf.A;
    N_inner_state_bits = (N-1)*k;
    
    
    N_status = 2^((N-1)*k);
    N_choices = 2^k;
    NextStat = zeros(N_status, N_choices);  % Status code always starts from 1.
    ConvOutput = cell(N_status, N_choices); % ConvOutput in binary form.
    
    if ~exist('soft_decode', 'var') || isempty(soft_decode)
        soft_decode=false;
    end
    
    for ns = 1:N_status
        for nc = 1:N_choices
            merged = [to_binary(nc-1,k),to_binary(ns-1,N_inner_state_bits)];
            NextStat(ns, nc) = from_binary(merged(1:N_inner_state_bits))+1;
            % calculate conv.
            o_block = false([1,n]);
            for o_iter=1:n
                o_block(o_iter) = logical(mod(merged*(A{o_iter}.'),2));
            end
            ConvOutput{ns, nc} = o_block;
        end
    end
    
    %% Viterbi decode. Hard decode.
    INF = length(encoded_bits);
    losses = INF*([0;ones(ns-1,1)]);
    if soft_decode
        n=1;
    end
    N_input_blocks = INF/n;
    best_path_choices = cell(N_input_blocks, 1);
    
    for iter=1:N_input_blocks
        best_path_choices{iter} = zeros(ns, 2);   % (choice, last_state)
    end
    
    for k_iter = 1:N_input_blocks
        index = n*(k_iter-1)+1;
        received_block = encoded_bits(:, index:index+n-1);
        new_losses = (NaN)*zeros(ns, 1);   % NaN means not updated.
        for ns = 1:N_status
            for nc = 1:N_choices
                o_block = ConvOutput{ns, nc};
                % Ensure that t_loss>=0 for correctness of Viterbi.
                if soft_decode
                    t_loss = received_block(from_binary(o_block)+1);
                else
                    t_loss = sum(xor(o_block, received_block));
                end
                t_ind = NextStat(ns, nc);
                if isnan(new_losses(t_ind))
                    new_losses(t_ind)=losses(ns)+t_loss;
                    best_path_choices{k_iter}(t_ind, 1) = nc-1;
                    best_path_choices{k_iter}(t_ind, 2) = ns;
                else
                    if new_losses(t_ind) > (losses(ns)+t_loss)
                        new_losses(t_ind) = losses(ns)+t_loss;
                        best_path_choices{k_iter}(t_ind, 1) = nc-1;
                        best_path_choices{k_iter}(t_ind, 2) = ns;
                    end
                end
            end
        end
        % update the best paths.
        losses = new_losses;
        % Notice: Here the storage of the decoded bits can be simplified by
        % pointers & linked lists in C++.
        % further running-time optimization needed.
    end
    
    %% Generate bit pedictions using best_path_choices.
    decoded_bits = false([1, N_input_blocks*k]);
    
    if conv_encoder_conf.trailing
        p=1;
    else
        % abrupt stopping.
        [~, p]=min(losses);
    end
    
    for iter=N_input_blocks:-1:1
        o_index = k*(iter-1)+1;
        status = best_path_choices{iter}(p, :);
        decoded_bits(o_index:o_index+k-1) = flip(to_binary(status(1),k));
        p = status(2);
    end
    
    if conv_encoder_conf.trailing
        decoded_bits = decoded_bits(1:N_input_blocks*k-N_inner_state_bits);
    end
end