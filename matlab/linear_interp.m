% Linear Interp
function est_ch = linear_interp(est_ch, pilot_flag)
    last_est = est_ch(1);
    last_est_idx = 1;
    for k=2:length(est_ch)
        if pilot_flag(k) == 1
            current_est = est_ch(k);
            linear_k = (current_est-last_est)/(k-last_est_idx);
            int_points = last_est + linear_k .* (1:k-last_est_idx-1);
            est_ch(last_est_idx+1: k-1) = int_points;
            est_ch(k) = est_ch(k);
            pilot_flag(k) = 1;
            last_est = current_est;
            last_est_idx = k;
        end
    end
end