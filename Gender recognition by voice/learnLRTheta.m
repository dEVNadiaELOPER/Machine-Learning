


function theta = learnLRTheta(X, y, lambda)
numLabels=1;
% LRClassifier returns the values of theta. Each row of theta corresponds
% to a single classifier for the number being considered.
% Some useful variables
m = size(X, 1); % number of examples
n = size(X, 2); % how many parameters (features)
theta = zeros(numLabels, n+1); % (n+1) to account for the x0 term
initialTheta = zeros(n+1,1);
options = optimset('GradObj','on','MaxIter',150); % used in fminunc
% Add ones to the X data matrix to account for x0
X = [ones(m, 1) X];


    yTemp = (y==1); % select all examples of particular number for training
    [tempTheta(:,1)] = fminunc(@(t)(cost(t,X,yTemp,lambda)),...
                               initialTheta,options);

    theta(1,:) = tempTheta(:,1)';
