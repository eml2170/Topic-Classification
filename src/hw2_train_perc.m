function params = hw2_train_perc(X,Y,num_passes)
d = 61188;

%initialize (w,theta)
w = zeros(1,d);
theta = 0;

n = numel(Y);
for i = 1:num_passes
   for j = 1:n 
      x = X(j,:);
      a = w*x' + theta;
      y = Y(j);
      if y*a <= 0
         %update perceptron weight vector
         w = bsxfun(@plus,w,y*x);
         theta = theta + y;
      end
   end
end
params = struct('w', w, 'theta', theta);
end