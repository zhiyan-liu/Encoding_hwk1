function est_ch = quad_interp(est_ch, pilot_flag, pilot_rate)
% est_ch为待插值的信道向量。其中有一些已经用训练符号估计了
% 其余值为0。（对应pilot_flag中的1和0）
% 此函数实现插值，将est_ch中为0的值补全。
    %idx=1:length(est_ch);
    %idx_before=idx(pilot_flag==1); 
    %est_ch_before=est_ch(pilot_flag==1);
    %est_ch=interp1(idx_before,est_ch_before,idx,'spline');
    
    %last_est = est_ch(1);
    last_est_idx = 1;
    k=2;
    while(k<=length(est_ch))
        pilot_num=0;
        while(pilot_num~=2 && k<=length(est_ch))
            if pilot_flag(k) == 1
                pilot_num=pilot_num+1;
            end
            k=k+1;
        end
        if(k~=length(est_ch))
            k=k-1;
        end
        current_est_idx = k;
        idx=last_est_idx:current_est_idx;
        est_ch_temp=est_ch(idx);
        pilot_flag_temp=pilot_flag(idx);
        est_ch_temp_2=est_ch_temp(pilot_flag_temp==1);
        temp_idx=idx(pilot_flag_temp==1);
        est_ch(idx)=interp1(temp_idx,est_ch_temp_2,idx,'spline');
        last_est_idx=current_est_idx;
        k=k+1;
    end    
end

