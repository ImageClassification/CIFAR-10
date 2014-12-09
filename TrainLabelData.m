function [trainXCs,M,P,trainXC_mean,trainXC_sd] = TrainLabelData(patch_size)

% Note: This is a customized version of the original code of the paper
% An Analysis of Single-Layer Networks in Unsupervised Feature Learning, Adam Coates, Honglak Lee, and Andrew Y. Ng. In AISTATS 14, 2011.
% which is found here:
% http://www.cs.stanford.edu/~acoates/papers/kmeans_demo.tgz 




%Load input data %
load('data.mat')


%patch_size = 6;
IMG_DIM = [32 32 3];
patch_limit = 10000; % Limit the number of patches to process
NUM_FEATURES = 10;

dictionary = zeros(patch_limit, patch_size*patch_size*3);


for i=1:patch_limit
    if(mod(i,2000)==0)
        fprintf('Processing patch: %d / %d\n', i, patch_limit); 
    end
    
    img = reshape(X_train(mod(i-1,size(X_train,1))+1, :), IMG_DIM); 
    patch_row = random('unid', IMG_DIM(1) - patch_size + 1);
    patch_col = random('unid', IMG_DIM(2) - patch_size + 1);
    patch = img(patch_row:patch_row+patch_size-1,patch_col:patch_col+patch_size-1,:); % Extracting a random patch
    dictionary(i,:) = patch(:)';
end

% Normalizing the patches

dictionary = (dictionary - repmat(mean(dictionary,2),[1,size(dictionary,2)]))./repmat(std(dictionary,[],2)+0.01,[1,size(dictionary,2)]);

[dictionary,M,P]=  whiten(dictionary); %whiten the extracted images

[IDX, centroids] = kmeans(dictionary, NUM_FEATURES, 'MaxIter',50,'EmptyAction','singleton');

trainXC = extract_features(X_train, centroids, patch_size, IMG_DIM, M,P);
trainXC_mean = mean(trainXC);
trainXC_sd = sqrt(var(trainXC)+0.01);
trainXCs = bsxfun(@rdivide, bsxfun(@minus, trainXC, trainXC_mean), trainXC_sd);
trainXCs = [trainXCs, ones(size(trainXCs,1),1)];



save('trainXCs.mat','trainXCs');
save('trainXC_mean.mat','trainXC_mean');
save('trainXC_sd.mat','trainXC_sd');
save('centroids.mat','centroids');
save('M.mat','M');
save('P.mat','P');



end