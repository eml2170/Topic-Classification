%implementation based on pseudocode on CML pg 48
function params = hw2_train_avgperc(X,Y,num_passes)
d = 61188;

%initialize
w = zeros(1,d); theta = 0;
u = zeros(1,d); beta = 0;

c = 1;

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
         
         %update cached weights and bias
         u = bsxfun(@plus,u,y*c*x);
         beta = beta + y*c;
      end
      c = c + 1;
   end
end
w = bsxfun(@minus,w,u/c); %average weights
theta = theta - beta/c; %average bias
params = struct('w', w, 'theta', theta);
end