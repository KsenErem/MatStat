pkg load interval

fid = fopen('Chanel.txt', 'r');
A = fscanf(fid, '%f', [200 1])
fclose(fid);

eps = z = zeros(200,1)
for i=1:1:size(eps)
  eps(i) = 10^-4
end
intervals = midrad(A(:, 1), 10^-4)

[oskorbin_center, k] = estimate_uncertainty_center(intervals)
oskorbin_center
