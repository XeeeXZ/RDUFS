function newL2 = bestMap(L1, L2)
% bestMap_simple: Approximate mapping from L2 to L1 using greedy matching.
% Does not require Hungarian algorithm.

L1 = L1(:);
L2 = L2(:);
if length(L1) ~= length(L2)
    error('L1 and L2 must be the same length.');
end

label1 = unique(L1);
label2 = unique(L2);
nClass1 = length(label1);
nClass2 = length(label2);

% Build confusion matrix
confMatrix = zeros(nClass1, nClass2);
for i = 1:nClass1
    for j = 1:nClass2
        confMatrix(i,j) = sum(L1 == label1(i) & L2 == label2(j));
    end
end

% Greedy matching: assign each L2 label to the L1 label with max overlap
mapping = zeros(nClass2,1);
used = false(nClass1,1);

for j = 1:nClass2
    [~, idx] = max(confMatrix(:,j));
    while used(idx)  % avoid duplicate assignments
        confMatrix(idx,j) = -1;
        [~, idx] = max(confMatrix(:,j));
    end
    mapping(j) = label1(idx);
    used(idx) = true;
end

% Apply mapping
newL2 = zeros(size(L2));
for j = 1:nClass2
    newL2(L2 == label2(j)) = mapping(j);
end
end
