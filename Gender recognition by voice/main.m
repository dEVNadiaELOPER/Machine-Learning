data = load('TrainData.csv');
dataTs = load('TestData.csv');
X = data(:,1:end-1);% features
y = data(:,end);% class labels
lambda_values=[0 0.001 0.003 0.01 0.03 0.1 0.3 13 10];
%[theta, lambda] = trainLRModel(x, y,lambda_values)

 
theta1 = learnLRTheta(X, y, 1);

x = normalizeFeatures(X);

[theta, lambda]=trainLRModel(X,y,lambda_values);
display('the lambda value is');
%display(theta1);
display(lambda);

Xs = dataTs(:,1:end-1);% features
ys = dataTs(:,end);% class labels
%Xs = normalizeFeatures(Xs);
Xss = [ones(size(Xs,1),1) , Xs];

yt= predictClass(Xss, theta', 0.5);
 [acc, recall, precision, fScore] = testPerformance(ys, yt)
 %%%%%%%%%%%%
 %BOUBS SVM
  display('svm');
SVMModel = fitcsvm(X,y);
label = predict(SVMModel,Xs);
[acc, recall, precision, fScore] = testPerformance(ys, label)
