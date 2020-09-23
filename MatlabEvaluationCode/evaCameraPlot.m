[exposure, SNRR, NOISER, SNRG, NOISEG, SNRB, NOISEB] = textread('/home/holo/Downloads/holo_camera/build/bin/EvaResualts.txt','%f %f %f %f %f %f %f');

figure;
plot(log(exposure), 20*log(NOISER),'r.');
hold on;
plot(log(exposure),  20*log(SNRR) , 'r*');
hold on;
plot(log(exposure),  20*log(NOISEG) , 'black.');
hold on;
plot(log(exposure),  20*log(SNRG) , 'black*');
hold on;
plot(log(exposure),  20*log(NOISEB) , 'blue.');
hold on;
plot(log(exposure),  20*log(SNRB) , 'blue*');



xlabel('Exposure,ms');
ylabel('20*log()');
title('SNR,NOISE');
legend('NOISE_R','SNR_R','NOISE_G','SNR_G','NOISE_B','SNR_B'); 


figure;
plot(exposure,  20*log(255 ./ SNRR) , 'ro');
hold on;
plot(exposure,  20*log(255 ./ SNRG) , 'blacko');
hold on;
plot(exposure,  20*log(255 ./ SNRB) , 'blueo');

xlabel('Exposure,ms');
ylabel('20*log()');
title('Dynamic Range');
legend('R','G','B'); 