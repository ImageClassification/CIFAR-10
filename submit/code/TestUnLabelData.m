function [testXCs]= TestUnLabelData(X_test,M,P,centroids,trainXC_mean,trainXC_sd,patch_size)


% Note: This is a customized version of the original code of the paper
% An Analysis of Single-Layer Networks in Unsupervised Feature Learning, Adam Coates, Honglak Lee, and Andrew Y. Ng. In AISTATS 14, 2011.
% which is found here:
% http://www.cs.stanford.edu/~acoates/papers/kmeans_demo.tgz 

IMG_DIM = [32 32 3];
testX = X_test;
testXC = extract_features(testX, centroids, patch_size, IMG_DIM, M,P);
testXCs = bsxfun(@rdivide, bsxfun(@minus, testXC, trainXC_mean), trainXC_sd);
testXCs = [testXCs, ones(size(testXCs,1),1)];


end