function [theta, J_history] = gradientDescent(m, X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

J_history = zeros(num_iters, 1);
J=0;
K=0;


for iter = 1:num_iters
   
    h=X*theta; %colums of first matrix must equal to rows of second matrix - m*1
    e=h-y; % mx1 - mx1 matric subtraction
    c=alpha*(1/m)*sum(e.*X); % mx1 vector multiply mxn vector 
    theta = theta - c';
    
     J_history(iter) = computeCost(m,X, y, theta);
end;

end
