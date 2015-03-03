function preds = hw2_test_bnb(params,test,Y) %Y is just a vector with all possible labels
    [m,~] = size(test); % get number of rows
    preds = zeros(m,1);
    for i = 1:m
        y_hat = hw2_test_classify(params, test(i,:),Y); %classify ith row
        preds(i,1) = y_hat;
    end
end

function y_hat = hw2_test_classify(params, x, Y)
    pi = params.pi;
    p1 = params.p1;
    p2 = params.p2;
    
    yn = numel(Y);
    y_values = zeros(1,yn); %vector of all likelihoods
    for k = 1:yn
        y_values(1,k) = eval(x,pi(k),p1(:,k),p2(:,k)); %return a scalar
    end
    y_hat = Y(find(y_values == max(y_values(:)),1)); %argmax
end

function l = eval(x, pi, p1, p2) %likelihood for a particular y
l = log(pi) + x*p1 + sum(p2);
end