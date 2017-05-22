%===================================================================
% Logistic Regression (Sigmoid Function) example
%
% Test Case:
%       Using exam1 and exam2 scores to predict college acceptance
%
%       The first two columns contains the exam scores and the third 
%       column of the dataset is the acceptance probability (1 = Yes, 
%       2 = No)
%===================================================================
%
% Initialization
clear ; close all; clc

% Load Data
data = load('ex2data1.txt');
X = data(:, [1, 2]); 
y = data(:, 3);

%===================================================================
% First Plot the sample data set
%===================================================================
fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);
figure; hold on;
pos = find(y==1); neg = find(y == 0);
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 8);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 8);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

%===================================================================
% Compute cost and gradient using sigmoid function
%===================================================================
%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

%===================================================================
% Optimize theta using fminunc
%===================================================================
%
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 

[theta, cost] = ...
	fminunc(@(t) (costFunction(t, X, y)) , initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

%===================================================================
% Optimize theta using Gradient Descent
%===================================================================
% 
num_iters = 100000;
alpha = 0.001;
theta2 = zeros(n+1,1);
grad2 = zeros(size(theta));
cost2 = 0;    

for iter = 1:num_iters
  cost2 = 0;
  grad2 = zeros(size(theta));
  predictions =  sigmoid(X*theta2);
  leftPart = -y' * log(predictions);
  rightPart = (1 - y)' * log(1 - predictions);
  cost2 = (1 / m) * (leftPart - rightPart);
  grad2 = (1/m) * X' * (predictions - y); %derivatives of cost
  theta2 = theta2 - alpha*grad2;
end;
	
% Print theta to screen
fprintf('Cost at theta found by Gradient Descent: %f\n', cost2);
fprintf('theta: \n');
fprintf(' %f \n', theta2);

%===================================================================
% Plot Boundary using each of the above thetas
%===================================================================
% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

% Plot Boundary
plotDecisionBoundary(theta2, X, y);
hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

%===================================================================
% Predict accuracy
%===================================================================
prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85 using fminuc, we predict an admission ' ...
         'probability of %f\n\n'], prob);
% Compute accuracy on our training set
p = round(sigmoid(X*theta));
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

prob = sigmoid([1 45 85] * theta2);
fprintf(['For a student with scores 45 and 85 using gradient descent, we predict an admission ' ...
         'probability of %f\n\n'], prob);
% Compute accuracy on our training set
p = round(sigmoid(X*theta2));
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

%===================================================================
% *** PART 2
%     use polynomial feature for non-linear dataset
%===================================================================
%% Initialization
fprintf('Regulartized Logistic Regression Test adding Polynomial Features\n');

data3 = load('ex2data2.txt');
X3 = data3(:, [1, 2]); y3 = data3(:, 3);

%===================================================================
% Plot sample data
%===================================================================
figure; hold on;
pos = find(y3==1); neg = find(y3 == 0);
plot(X3(pos, 1), X3(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 8);
plot(X3(neg, 1), X3(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 8);
hold on;
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
% Specified in plot order
legend('y = 1', 'y = 0')
hold off;

%===================================================================
% Regularized Logistic Regression - Add Polynomial Features
% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled. 
% Try Lambda = 1, 0.1, 10, 100 (lambda of 1 is best result)
%===================================================================
X3 = mapFeature(X3(:,1), X3(:,2));

% Initialize fitting parameters
initial_theta3 = zeros(size(X3, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost3, grad3] = costFunctionReg(initial_theta3, X3, y3, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost3);

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta3, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X3, y3, lambda)), initial_theta3, options);

%===================================================================
% Plot Boundary
%===================================================================
plotDecisionBoundary(theta3, X3, y3);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

%===================================================================
% Test output
%===================================================================
testX1 = 0.20104;
testX2 = -0.60161;
testX = mapFeature(testX1, testX2); % output is 1
prob = round(sigmoid(testX*theta3));
fprintf(['For a Microchip test with scores %f and %f, we predict an a=output ' ...
         'probability of %f\n'], testX1, testX2, prob);
    
testX1 = -0.4038;
testX2 = -0.70687;
testX = mapFeature(testX1, testX2); % output is 0
prob = round(sigmoid(testX*theta3));
fprintf(['For a Microchip test with scores %f and %f, we predict an a=output ' ...
         'probability of %f\n'], testX1, testX2, prob);
         
% Compute accuracy on our training set
p = round(sigmoid(X3*theta3));
fprintf('Train Accuracy using Lambda = %f: %f\n', lambda, mean(double(p == y3)) * 100);

