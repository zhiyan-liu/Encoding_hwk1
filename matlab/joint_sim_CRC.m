setup_mapper;
setup_encoder;

%% Simulation parameters.
N_sim = 500;
N_info_bits = 4096;
SNR_arr = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10, 12.5, 15]; % target SNR.
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

tic;
for sigma_iter = 1:N_sigmas
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
        pred_bits = bit_demapping(syms_with_noise, length(encoded_bits), mapping_conf, ch);
        
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
        for k=1:16:length(decoded_validation)
            if(any(decoded_validation(k:k+15)))
                err_box_cnt_crc(sigma_iter)=err_box_cnt_crc(sigma_iter)+1;
            end
        end
        
       %%
        if mod(sim_iter, floor(N_sim/10))==0
            disp(['SNR=', num2str(SNR_arr(sigma_iter)),': ',num2str(sim_iter/N_sim*100),'% complete']);
        end
        
    end
    err_box_cnt_crc(sigma_iter)=err_box_cnt_crc(sigma_iter)/...
        (ceil(length(decoded_validation)/16)*N_sim);
    
    disp([num2str(sigma_iter),'/', num2str(N_sigmas),' Complete for SNR=',...
        num2str(SNR_arr(sigma_iter)), 'dB']);
    if ~soft_decode
        disp(['Log BER before encoding: ', ...
            num2str(log10(err_bit_cnt_before_coding(sigma_iter)/(L_encoded(sigma_iter)*N_sim)))]);
    end
    disp(['Log BER after encoding: ', ...
        num2str(log10(err_bit_cnt_after_coding(sigma_iter)/(N_info_bits*N_sim)))]);
end
time_elapsed = toc;
assert(~any(diff(L_encoded)), 'error in length of encoded bits!');

%% Count BLER.
figure(1);
plot(SNR_arr,err_box_cnt_crc);
ylabel('err Box\_rate');
xlabel('SNR_(_d_B_)');



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

%% Save files.
save('data/sim.mat');