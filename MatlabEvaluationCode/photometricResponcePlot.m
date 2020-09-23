[u]  = load('/home/holo/Downloads/holo_camera/build/bin/photoCalibResult/pcalib.txt');
i = 0:255;figure ;plot(i,log(u(i+1)),'.');
title('Photometric Response Function');
xlabel('Intensity');
ylabel('Ln(E*deltaT');

i = 0:255;figure ;plot(i,(u(i+1)),'.');
title('Photometric Response Function');
xlabel('Intensity');
ylabel('E*T');

i = 1:255;figure ;plot(i,log(u(i+1)) - log(u(i)),'.');
title('Photometric Response Function');
xlabel('Intensity');
ylabel('Ln(E*deltaT');