function L2SVM(filename,X_train,y_train,X_test,opt)


% Note: This is a customized version of the original code of the paper
% An Analysis of Single-Layer Networks in Unsupervised Feature Learning, Adam Coates, Honglak Lee, and Andrew Y. Ng. In AISTATS 14, 2011.
% which is found here:
% http://www.cs.stanford.edu/~acoates/papers/kmeans_demo.tgz 

addpath(genpath('./minFunc/'));
addpath ./tinyclassifier/    
addpath ./helpers
y_train = double(y_train);


params = trainClassifier(X_train(1:end,:), y_train(1:end), opt);
preds = predictClassifier(params, X_train);
fprintf('Train Accuracy = %.2f%%\n', 100*mean(preds(:) == y_train(:)));

preds = predictClassifier(params, X_test);


% write the data out to a file that can be read by Kaggle.
writeLabels(filename, preds);
end