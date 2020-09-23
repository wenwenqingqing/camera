alpha = 1;d=0.3;
figure;
Kp = 0.1;
    gamma = 0.5:0.05:2;
    f = 1 + alpha*Kp*(d*tan((1-gamma)*atan(1/d)));
    
    hold on;
    plot((gamma),f,'black.');

Kp = 0.4;
    gamma = 0.5:0.05:2;
    f = 1 + alpha*Kp*(d*tan((1-gamma)*atan(1/d)));
    
    hold on;
    plot((gamma),f,'black--');
    
    Kp = 0.7;
    gamma = 0.5:0.05:2;
    f = 1 + alpha*Kp*(d*tan((1-gamma)*atan(1/d)));
    
    hold on;
    plot((gamma),f,'black*');
    
    Kp = 1.0;
    gamma = 0.5:0.05:2;
    f = 1 + alpha*Kp*(d*tan((1-gamma)*atan(1/d)));
    
    hold on;
    plot((gamma),f,'black-');
    
title('d = 0.3');
legend('Kp=0.1','Kp=0.4','Kp=0.7','Kp=1.0');
xlabel('\gamma');
ylabel('{t_{i+1}}/{t_i}')

alpha = 1; Kp=1.0;

figure;
d = 0.1;
gamma = 0.5:0.05:2;
f = 1 + alpha*Kp*(d*tan((1-gamma)*atan(1/d)));
    
hold on;
plot((gamma),f,'black.');

d = 0.4;
gamma = 0.5:0.05:2;
f = 1 + alpha*Kp*(d*tan((1-gamma)*atan(1/d)));
    
hold on;
plot((gamma),f,'black--');hold on;

d = 0.7;
gamma = 0.5:0.05:2;
f = 1 + alpha*Kp*(d*tan((1-gamma)*atan(1/d)));
    
hold on;
plot((gamma),f,'black*');hold on;


d = 1.0;
gamma = 0.5:0.05:2;
f = 1 + alpha*Kp*(d*tan((1-gamma)*atan(1/d)));
    
hold on;
plot((gamma),f,'black-');hold on;



title('Kp = 1.0');
legend('d=0.1','d=0.4','d=0.7','d=1.0');
xlabel('\gamma');
ylabel('{t_{i+1}}/{t_i}')