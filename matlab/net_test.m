%%preprocessing
uvd_pre = zeros(1,3);

loadfile = 'data.mat';
load(loadfile);
loadfile = 'target.mat';
load(loadfile);

uvd = data(:,100)';
xyz = target(:,100)';

uvd_pre(1,1) = (uvd(1,1)-2) / (511-2);
uvd_pre(1,2) = uvd(1,2) / 423;
uvd_pre(1,3) = uvd(1,3) / 1000;

%Layer 1
out1 = uvd_pre * W1' + b1';
for i = 1:32
    if out1(1,i) < 0
        out1(1,i) = 0;
    end
end

%Layer 2
out2 = out1 * W2' + b2';
for i = 1:32
    if out2(1,i) < 0
        out2(1,i) = 0;
    end
end

%Layer 3
out3 = out2 * W3' + b3';
for i = 1:32
    if out3(1,i) < 0
            out3(1,i) = 0;
    end
end

%Layer 4
out4 = out3 * W4' + b4';
%for i = 1:3
%    if out4(1,i) < 0
%        out4(1,i) = 0;
%   end
%end

out4
xyz

error = sqrt( (out4(1,1)-xyz(1,1))^2 + (out4(1,2)-xyz(1,2))^2 + (out4(1,3) - xyz(1,3))^2 )