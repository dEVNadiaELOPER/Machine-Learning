function [theta, lambda] = trainLRModel(x, y,lambda_values)
%theta = learnLRTheta(x, y,1);
min=1111111111111111;%LARGE VALUE
minL=0;% to save min lambda

for k= 1:size(lambda_values') 
     sumTs=0;%test err sum for each lambda
    sumTr=0;%training err sum for each lambda
  
for j = 1:10
   %clean  Y X training and testing every itraition
    XTs = [];
     yTs=[];
     XTr=[];
      yTr=[];   
    counT=1;
    counTs=1;
  indices = crossvalind('Kfold',y,10);%cross validation
    for i = 1:2376
        %if indices(i)==j then add the object to test set
        if indices(i)==j
         XTs(counTs,:)=x(i,:);
         yTs(counTs,:)=y(i,:);   
         counTs=counTs+1;
          %else indices(i)!=j then add the object to training set
        else
       XTr(counT,:)=x(i,:);
         yTr(counT,:)=y(i,:);   
        counT=counT+1;
        end
        
     
    end
   
%learn Thaeta for lambda k
theta1 = learnLRTheta(XTr, yTr, lambda_values(k));
XTr = [ones(size(XTr,1),1) , XTr];
 err = cost(theta1',XTr, yTr, 0);
 sumTr=sumTr+err;
XTs = [ones(size(XTs,1),1) , XTs];
 errTs = cost(theta1',XTs, yTs, 0);
 sumTs=sumTs+errTs;

 
end
%error for training and validation 
errA(k)=sumTr/10
errTsA(k)=sumTs/10;
if errTsA(k)<min
 min=errTsA(k);
 %to save min lambda and its theta
 minL=lambda_values(k);

theta=theta1;


end
end

lambda=minL;
theta = learnLRTheta(x, y, lambda);
%plot training and validation errors Vs.lambda values curve

figure;
plot(lambda_values,errTsA,lambda_values,errA);
ylabel('Cost') 
xlabel('lambda')
legend({'y = validation','y = training'},'Location','southwest')


end
