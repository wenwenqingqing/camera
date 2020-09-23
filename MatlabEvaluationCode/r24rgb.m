close all;clear all;
%[r,g,b,rn,gn,bn] = textread('/home/holo/Downloads/calib_ar0231/24rgb1.txt','%f,%f,%f,%f,%f,%f');
[r,g,b,rn,gn,bn] = textread('/home/holo/Downloads/m19404/ColorConstancy/24rgb/24rgb4.txt','%f,%f,%f,%f,%f,%f');

figure;
plot(r,'r');
hold on;
plot(g,'g');
hold on;
plot(b,'b');

figure;
plot(rn,'r');
hold on;
plot(gn,'g');
hold on;
plot(bn,'b');
legend('noise r','noise g','noise b');
title('24 kind rgb noise');
xlabel('rgb index');ylabel('noise');

X = sprintf('R NOISE sqrt(var) : %f', sqrt(var(rn)));
disp(X);
X = sprintf('G NOISE sqrt(var) : %f', sqrt(var(gn)));
disp(X);
X = sprintf('B NOISE sqrt(var) : %f', sqrt(var(bn)));
disp(X);

ssRGB = [128.4,84.57,72.99, 166.4,142.8,50.32, 59.19,60.10,150.3, 232.2,234.5,234.3, ...
         213.9,164.7,142.3, 67.38,93.98,168.9, 72.24,141.7,76.29, 211.2,214.6,211.1, ...
         76.76,128.1,161.3, 164.4,102.7,111.4, 162.5,48.34,46.70, 166.4,167.5,168.2, ...
         89.51,104.7,67.13, 79.12,58.66,110.1, 193.3,202.9,21.97, 129.3,129.8,131.5, ...
         145.5,163.4,183.5, 174.8,199.8,76.04, 173.3,93.30,165.6, 89.55,89.15,90.88, ...
         128.7,216.5,191.5, 205.1,175.4,33.10, 3.634,133.0,166.6, 57.03,57.27,56.35];

for j = 1 %0.1:0.1:1.9
    sRGB = [];
    rgb = [];

    H = [];
    Hi = [];
    B = [];
    for i = 1:72
        linearRGB = 255*(ssRGB(i)/255)^j;
        if linearRGB > 255
            linearRGB = 255;
        end
        sRGB = [sRGB linearRGB];
    end
    for i = 1:24
        rgbi = [ r(i) g(i) b(i)]';
        rgb = [rgb ; rgbi];
%         Hi = [r(i),g(i),b(i),0,0,0,0,0,0,1,0,0;
%               0,0,0,r(i),g(i),b(i),0,0,0,0,1,0;
%               0,0,0,0,0,0,r(i),g(i),b(i),0,0,1];
        Hi = [r(i),g(i),b(i),0,0,0,0,0,0,1,0,0;
              0,0,0,r(i),g(i),b(i),0,0,0,0,1,0;
              0,0,0,0,0,0,r(i),g(i),b(i),0,0,1];
        H = [H ; Hi];
    end

    A = inv(H'*H)*H'*sRGB';
    for i = 1:24
         Bi = [A(10) A(11) A(12)];
         B = [B ; Bi'];
    end

    error = sRGB' - H*A;
    b_c = [];
    r_c = [];
    g_c = [];
    r_r = [];
    g_r = [];
    b_r = [];
    
    ssr = [];
    ssg = [];
    ssb = [];
    
    for i = 1:24
        tmp = r(i)*A(1)+g(i)*A(2)+b(i)*A(3) + A(10);
        if tmp > 255
            tmp =255;
        end
        r_c = [r_c 255*(tmp/255)^(1/j)];
        tmp = r(i)*A(4)+g(i)*A(5)+b(i)*A(6) + A(11);
        if tmp > 255
            tmp =255;
        end
        g_c = [g_c 255*(tmp/255)^(1/j)];
        tmp = r(i)*A(7)+g(i)*A(8)+b(i)*A(9) + A(12);
        if tmp > 255
            tmp =255;
        end
        b_c = [b_c 255*(tmp/255)^(1/j)];
        
        r_r = [r_r sRGB(i*3 -2)];  
        g_r = [g_r sRGB(i*3 -1)];
        b_r = [b_r sRGB(i*3)];
        
        ssr = [ssr ssRGB(i*3-2)];
        ssg = [ssg ssRGB(i*3 -1)];
        ssb = [ssb ssRGB(i*3)];
    end
    figure;
    plot(r_c,'r');
    hold on;
    plot(r_r,'black');
    hold on;
    diff = r_c-r_r;
    hold on;
    plot(diff,'y');
    legend('RGB correction','sRGB','diff');
    title('R correction CALIBRAION');
    xlabel('Color Index');
    xlabel('COLOR');
    X = sprintf('R sqrt(var) : %f', sqrt(var(diff)));
    disp(X);
    
    figure;
    plot(g_c,'g');
    hold on;
    plot(g_r,'black');
    hold on;
    diff = g_c-g_r;
    hold on;
    plot(diff,'y');
    legend('RGB correction','sRGB','diff');
    title('G correction CALIBRAION');
    xlabel('Color Index');
    xlabel('COLOR');
    X = sprintf('G sqrt(var) : %f', sqrt(var(diff)));
    disp(X);
    
    figure;
    plot(b_c,'blue');
    hold on;
    plot(b_r,'black');
    hold on;
    diff = b_c-b_r;
    hold on;
    plot(diff,'y');
    legend('RGB correction','sRGB','diff');
    title('B correction CALIBRAION');
    xlabel('Color Index');
    xlabel('COLOR');
    X = sprintf('B sqrt(var) : %f', sqrt(var(diff)));
    disp(X);
    
    % error0 = sRGB' - rgb;
    rgbHB = (H*A);
    rgbHB = 255*(rgbHB/255).^(1/j);
    error0 = rgbHB - ssRGB';
    figure;
    plot(error,'r');
    hold on;
    plot(error0,'g');
    X = sprintf('ssRGB sqrt(var) : %f', sqrt(var(error0)));
    disp(X);

end

diff = r_c -ssr;
figure;plot(diff,'y');hold on;plot(r_c,'r');hold on;plot(ssr,'black');hold on;plot(r,'cyan');
title('R');legend('diff','correction','sRGB','measurement');xlabel('index');ylabel('color');
    X = sprintf('R sqrt(var) : %f', sqrt(var(diff)));
    disp(X);
diff = g_c -ssg;
figure;plot(diff,'y');hold on;plot(g_c,'g');hold on;plot(ssg,'black');hold on;plot(g,'cyan');
title('G');legend('diff','correction','sRGB','measurement');xlabel('index');ylabel('color');
    X = sprintf('G sqrt(var) : %f', sqrt(var(diff)));
    disp(X);
diff = b_c -ssb;
figure;plot(diff,'y');hold on;plot(b_c,'blue');hold on;plot(ssb,'black');hold on;plot(b,'cyan');
title('B');legend('diff','correction','sRGB','measurement');xlabel('index');ylabel('color');
    X = sprintf('B sqrt(var) : %f', sqrt(var(diff)));
    disp(X);
   

rw = [];
gw = [];
bw = [];

rwo = [];
gwo = [];
bwo = [];

rwc = [];
gwc = [];
bwc = [];

for i = 1:24
    if mod(i, 4) == 0
        rw = [rw r(i)];
        gw = [gw g(i)];
        bw = [bw b(i)];
        
        rwc = [rwc rgbHB(i*3-2)];
        gwc = [gwc rgbHB(i*3-1)];
        bwc = [bwc rgbHB(i*3-0)];
        
        rwo = [rwo ssRGB(i*3-2)];
        gwo = [gwo ssRGB(i*3-1)];
        bwo = [bwo ssRGB(i*3-0)];
    end
    r(i) = 1.27*1.03*r(i);
    g(i) = 1.03*g(i);
    b(i) = 1.03*b(i);
end


figure;
plot(rw,'r');
hold on;
plot(gw,'green');
hold on;
plot(bw,'blue');legend('r','g','b');
title('origin low color temperature light source');
%Robut Automatic White Balance Algorithm using Gray Color Points in Images
figure;
plot(rwo,'r');
hold on;
plot(gwo,'green');
hold on;
plot(bwo,'blue');
title('standard GRAY');
figure;
plot(rwc,'r');
hold on;
plot(gwc,'green');
hold on;
plot(bwc,'blue');
legend('rc','gc','bc');
title('COR GRAY');

% r_m = mean(r);g_m = mean(g); b_m = mean(b);
% rgb_m = (r_m + g_m + b_m) / 3;
% G_r = rgb_m / r_m;
% G_g = rgb_m / g_m;
% G_b = rgb_m / b_m;
