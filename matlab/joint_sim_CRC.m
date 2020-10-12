setup_mapper;
setup_encoder;

%% Simulation parameters.
N_sim = 1000;
N_info_bits = 4096;
SNR_arr = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 12.5, 15, 17.5];   % target SNR.
Ps = 1;


%% Start simulation.
SNRs_abs = 10.^(SNR_arr/10);
sigma_arr = sqrt(Ps./SNRs_abs);
N_sigmas = length(sigma_arr);

err_bit_cnt_after_coding = zeros(N_sigmas, 1);
err_bit_cnt_before_coding = zeros(N_sigmas, 1);
err_box_cnt_crc = zeros(N_sigmas,1);
soft_decode = false;
L_encoded = zeros(N_sigmas, 1);

if soft_decode
    mapping_conf.out = 'L2';
else
    mapping_conf.out = 'hard';
end

% Record all the channel CSI and Constellation points.
record_csi = false;
CSI = cell(N_sigmas, 1);    % each row of each cell element: The ch.
SYMS_TRANSMIT = cell(N_sigmas, 1);
SYMS_RECEIVE = cell(N_sigmas, 1);

tic;
parfor sigma_iter = 1:N_sigmas
    sigma = sigma_arr(sigma_iter);
    
    for sim_iter = 1:N_sim
        random_bits = (rand([1, N_info_bits])>0.5);
        random_bits_with_crc = CRC(random_bits);
        encoded_bits = conv_encode(random_bits_with_crc, conv_encoder_conf);
        
        %% Setup length of encoded bits.
        L_encoded(sigma_iter) = length(encoded_bits);
        
        %% Simulation with channel.
        syms = bit_mapping(encoded_bits, mapping_conf);
        ch = ch_realization(length(syms), ch_conf);
        syms_with_noise = syms .* ch + (get_cgaussian(sigma, length(syms))).';
        if record_csi
            if isempty(CSI{sigma_iter})
                Ls = length(syms);
                CSI{sigma_iter} = zeros(N_sim, Ls);
                SYMS_TRANSMIT{sigma_iter} = zeros(N_sim, Ls);
                SYMS_RECEIVE{sigma_iter} = zeros(N_sim, Ls);
            end
            CSI{sigma_iter}(sim_iter, :) = ch;
            SYMS_TRANSMIT{sigma_iter}(sim_iter, :) = syms;
            SYMS_RECEIVE{sigma_iter}(sim_iter, :) = syms_with_noise;
        end
        pred_bits = bit_demapping(syms_with_noise, length(encoded_bits), mapping_conf, ch, ch_conf, sigma);
        
        if ~soft_decode
            err_bit_cnt_before_coding(sigma_iter) = err_bit_cnt_before_coding(sigma_iter) + ...
                sum(xor(pred_bits, encoded_bits));
        end
        
        %% Decode.
        decoded_bits_with_crc = fast_conv_decode(pred_bits, conv_encoder_conf, soft_decode);
        decoded_validation = deCRC(decoded_bits_with_crc);
        
        %% Find all the errors.       
        err_bit_cnt_after_coding(sigma_iter) = err_bit_cnt_after_coding(sigma_iter) + ...
        sum(xor(random_bits_with_crc, decoded_bits_with_crc));
        for k=1:length(decoded_validation(:,1))
            if(any(decoded_validation(k, :)))
                err_box_cnt_crc(sigma_iter)=err_box_cnt_crc(sigma_iter)+1;
            end
        end
        
       % Display running info.
        if mod(sim_iter, floor(N_sim/10))==0
            disp(['SNR=', num2str(SNR_arr(sigma_iter)),': ',num2str(sim_iter/N_sim*100),'% complete']);
        end
        
    end
    err_box_cnt_crc(sigma_iter)=err_box_cnt_crc(sigma_iter)/...
        (ceil(numel(decoded_validation)/16)*N_sim);
    
    disp([num2str(sigma_iter),'/', num2str(N_sigmas),' Complete for SNR=',...
        num2str(SNR_arr(sigma_iter)), 'dB']);
    if ~soft_decode
        disp(['Log BER before encoding: ', ...
            num2str(log10(err_bit_cnt_before_coding(sigma_iter)/(L_encoded(sigma_iter)*N_sim)))]);
    end
    disp(['Log BER after encoding: ', ...
        num2str(log10(err_bit_cnt_after_coding(sigma_iter)/(N_info_bits*N_sim)))]);
    disp(['Log BLER after encoding: ', ...
        num2str(log10(err_box_cnt_crc(sigma_iter)))]);
    
end
time_elapsed = toc;
assert(~any(diff(L_encoded)), 'error in length of encoded bits!');


%% Count BLER.
figure(1);
set(gca, 'yscale', 'log');
plot(SNR_arr,err_box_cnt_crc);
ylabel('BLER');
xlabel('SNR_(_d_B_)');
grid on;

%% Display!!
figure(2);
hold on;
set(gca, 'yscale', 'log');
plot(SNR_arr, ((err_bit_cnt_after_coding)/(N_info_bits*N_sim)).');
plot(SNR_arr, (err_bit_cnt_before_coding/(L_encoded(1)*N_sim)).');
legend('BER_ conv', 'BER_ ch');
title('BER-SNR Curve');
xlabel('SNR(dB)');
ylabel('BER');
grid on;

disp(['Time elapsed: ', num2str(time_elapsed), 's for ', num2str(N_sigmas*N_sim), ' channel simulations']);
disp(['b=', num2str(ch_conf.b), ', rho=',num2str(ch_conf.rho)]);

%% Save variables into files.
save(['data/sim_', strrep(datestr(datetime), ':', '_'), '.mat']);
