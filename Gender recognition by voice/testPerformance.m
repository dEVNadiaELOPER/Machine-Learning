function [acc, recall, precision, fScore] = testPerformance(y, y_predicted)
%build the confusion matrix using calError method
[tp, fp, tn, fn] = calError(y, y_predicted);
p=tp+fn;
n=tn+fp;
N=tn+fp+tp+fn;
tp_rate = tp/p;
tn_rate = tn/n;
%Calculates accuracy,recall,precision,and f-score.
acc = (tp+tn)/N;
sensitivity = tp_rate;
specificity = tn_rate;
precision = tp/(tp+fp);
recall = sensitivity;
fScore = 2*((precision*recall)/(precision + recall));

