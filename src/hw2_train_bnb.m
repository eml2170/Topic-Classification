function params = hw2_train_bnb(X,Y,possibleY)

%range of k (number of classes)
%yn = 20; 
yn = numel(possibleY);

pi = zeros(1,yn); %initialize pi
n = numel(Y); %number of points

for i = 1:yn
    pi(i) = sum(Y == possibleY(i))/n; %compute pi_y, which is just the fraction of observed labels y
end

d = 61188; %range of j

p1 = zeros(d,yn); %initialize p1 and p2, which are functions of p_j,k.
p2 = zeros(d,yn);
for k = 1:yn
    subdata = X(Y == possibleY(k),:);
    [m,~] = size(subdata);
    denom = 2 + m; %laplace smoothing
    for j = 1:d
        num = 1 + sum(subdata(:,j) == 1); %laplace smoothing
        p=num/denom; %compute p_j,k
        p1(j,k) = log(p/(1-p));
        p2(j,k) = log(1-p);
    end
end
params = struct('pi',pi,'p1',p1,'p2', p2);
end