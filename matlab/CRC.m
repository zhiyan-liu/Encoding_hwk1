function [ signal_CRC ] = CRC( signal )
%input signal£º original signal
%output signal_CRC£ºthe signal encoded
signal=signal.';
g=[1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1];
G=length(g);
for k=1:80:length(signal)
    signal_part=signal(k:min(k+79,end));
    a=[signal_part;zeros(G-1,1)];
    [~,CRC_code]=deconv(a,g);
    CRC_code=mod(CRC_code(end-G+2:end),2);
    signal_CRC_part=[signal_part;CRC_code];
    if(k==1)
        signal_CRC=signal_CRC_part;
    else
        signal_CRC=[signal_CRC;signal_CRC_part];
    end
end
signal_CRC=signal_CRC.';
end