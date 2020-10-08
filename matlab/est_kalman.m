function est_ch = est_kalman(est_ch, pilot_flag, R, rho, b, sigma)
a = rho ^ R;
var_u = (1-rho^2) * sum(rho.^(0:R-1));
var_s = 1;
var_n = sigma^2 / b^2;
x = (est_ch(pilot_flag) - sqrt(1-b^2)) / b;
x(end) = [];
s = zeros(1,length(x));
M = zeros(1,length(x));
K = zeros(1,length(x));
s(1) = 0;
M(1) = var_s;
for k = 1:length(x)
    if k >= 2
        s(k) = a * s(k-1);
        M(k) = a^2 * M(k-1) + var_u;
    end
    K(k) = M(k) / (M(k) + var_n);
    s(k) = s(k) + K(k) * (x(k) - s(k));
    M(k) = (1 - K(k)) * M(k);
    if k >= 2
        s_last = s(k-1);
        s_now = s(k);
        factor = (s_now/s_last)^(1/R);
        pred_s = factor.^(1:R-1) * s_last;
        pred_ch = sqrt(1-b^2) + b*pred_s;
        est_ch((k-2)*R+2:min((k-1)*R,length(est_ch)-1)) = pred_ch(1:1+(min((k-1)*R,length(est_ch)-1)-(k-2)*R-2));
    end
    if k == length(x)
        pred_s = rho.^(1:R-1) * s(k);
        pred_ch = sqrt(1-b^2) + b*pred_s;
        est_ch((k-1)*R+2:min(k*R,length(est_ch)-1)) = pred_ch(1:1+(min(k*R,length(est_ch)-1)-(k-1)*R-2));
    end
end
% figure;
% plot(real(x(1:500)));
% hold on;
% plot(real(s(1:500)));
% real_s = (ch(pilot_flag) - sqrt(1-b^2)) / b;
% real_s(end) = [];
% hold on;
% plot(real(real_s(1:500)));
% legend('observe','pred','real');
% mean(abs(x-real_s).^2)
% mean(abs(real_s-s).^2)
end

