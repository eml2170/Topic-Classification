function preds = hw2_test_perc(params, test)
    [n,~] = size(test);
    preds = zeros(n,1);
    for i=1:n
        x = test(i,:);
        preds(i) = perc_classify(params,x);
    end
end

function result = perc_classify(params,x)
   w = params.w;
   theta = params.theta;
   
   result = sign(w*x' + theta);
end