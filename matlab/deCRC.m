function [ signal ] = deCRC( receiveCRC )
%input receiveCRC£ºsignal received
%output signal£ºthe signal decoded
g=[1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1];
G=length(g);
for k=1:96:length(receiveCRC)
    signal_part=receiveCRC(k:min(k+95,end));
    [~,part]=deconv(signal_part,g);
    part=mod(part(end-G+2:end),2);
    if(k==1)
        signal=part;
    else
        signal=[signal;part];
    end
end
end
