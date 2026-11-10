function D = EuDist2(fea_a, fea_b, bSqrt)
% EuDist2 - Efficient Euclidean Distance Computation
% D = EuDist2(fea_a, fea_b, bSqrt)
% 
% fea_a: n1 x d matrix
% fea_b: n2 x d matrix (optional, default = fea_a)
% bSqrt: whether to take square root (optional, default = 1)

if nargin < 2 || isempty(fea_b)
    fea_b = fea_a;
end

if nargin < 3
    bSqrt = 1;
end

AA = sum(fea_a.^2, 2);
BB = sum(fea_b.^2, 2);
AB = fea_a * fea_b';

D = bsxfun(@plus, AA, BB') - 2 * AB;
D(D < 0) = 0; % 由于数值误差，可能会出现小于0的值

if bSqrt
    D = sqrt(D);
end
end