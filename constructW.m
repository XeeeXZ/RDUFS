function W = constructW(fea, options)
% 构建KNN热核图相似度矩阵
% fea: n x d 数据矩阵
% options.k: KNN数量
% options.t: 热核宽度参数

[nSmp, ~] = size(fea);
D = EuDist2(fea);  % 欧几里得距离
[~, idx] = sort(D, 2);

k = options.k;
t = options.t;

W = zeros(nSmp);
for i = 1:nSmp
    for j = 2:k+1 % idx(i,1) 是自己
        id = idx(i,j);
        W(i,id) = exp(-D(i,id)^2 / (2*t^2));
        W(id,i) = W(i,id); % 保证对称
    end
end
end
