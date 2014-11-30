function [clusterBelonging] = SeededKMeans()

load('data.mat')
K = 10;
N_Comp = 5;
pca_ceoff = zeros(size(X_train,2),N_Comp,K);
%train_pca = zeros(size(X_train,1),100,K);
test_pca = zeros(size(X_train,1),N_Comp,K);
mean_PCA = zeros(1,N_Comp,K);
len_PCA = zeros(K,1);

for i = 1:K
   classX = X_train(find(y_train==i),:);
   minX = repmat(min(classX,[],2),[1,size(classX,2)]);
   maxX = repmat(max(classX,[],2),[1,size(classX,2)]);
   classX = (classX - minX)./(maxX - minX);
   meanX = repmat(mean(classX,2),[1,size(classX,2)]);
   classX = classX - meanX;
   [coeff,score,latent]  = pca(classX,'NumComponents',N_Comp);
   pca_ceoff(:,:,i) = coeff;   
   mean_PCA(:,:,i) = mean(classX*coeff,1);
   len_PCA(i,1) = size(classX,1);
end



classX = X_train;
minX = repmat(min(classX,[],2),[1,size(classX,2)]);
maxX = repmat(max(classX,[],2),[1,size(classX,2)]);
classX = (classX - minX)./(maxX - minX);
meanX = repmat(mean(classX,2),[1,size(classX,2)]);
classX = classX - meanX;
for i = 1:K
   test_pca(:,:,i) = classX*pca_ceoff(:,:,i); %size of test set x 100 x 10(classes)
end



N = size(classX,1);


clusterBelonging = zeros(N,1);
clusterCenters = mean_PCA;% 1 100 10

%dataKDim = repmat(data, [1 1 K]);

while 1
    clusterCentersK = repmat(clusterCenters,[size(test_pca,1),1,1]);
    dist = test_pca - clusterCentersK;
    euDist = sqrt(sum(dist.^2,2));
    [val,clusterBelongingNew] = min(euDist,[],3);
    if clusterBelongingNew == clusterBelonging;
        break;
    else
        clusterBelonging = clusterBelongingNew;
    end
    
    for i=1:K
        findI = find(clusterBelonging==i);
        if findI
            unlabeled = test_pca(find(clusterBelonging==i),:,i);
            clusterCenters(:,:,i) = (sum(unlabeled,1) +mean_PCA(:,:,i).*len_PCA(i,1))./(size(unlabeled,1)+len_PCA(i,1)) ;
        end
    end
    
end

mean(y_train ~= clusterBelonging)
%writeLabels('SeedKMean.csv', clusterBelonging);

end