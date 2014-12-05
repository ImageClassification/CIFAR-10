function runClassifier(dataset, opt)
%RUNCLASSIFIER Runs a simple SVM or MLR classifier.
% dataset - either 'random' or './path/to/dataset/' containing
%           entries X_train, X_test, y_train, (y_test - optional).
% opt     - options to run with:
%     .loss     - 'mlr' for softmax regression and 'l2svm' for L2 SVM.
%                 Default is 'mlr'.
%
%     .lambda   - regularization parameter. Default is 0.
%
%     .dual     - optimize in the dual if true. Default is false. If false
%                 then a linear kernel is used.
%
%     .kernelfn - kernel function - Either a string 'rbf' for RBF kernel or
%                 'poly' for a polynomial kernel.
%                 Alternatively, kernelfn can be a function kernelfn(x, y)
%                 which should return an m1 x m2 gram matrix between 
%                 x and y, where there are m1 examples in x and m2 in y.
%                 For example you can implement a tanh kernel with params
%                 a and b as opt.kernelfn = @(X1, X2) tanh(a*X1*X2' - b).
%                 Default is 'rbf'.
%
%     .gamma    - RBF kernel width. Larger gamma => smaller variance.
%                 gaussian. Default is 1.
%
%     .order    - Polynomial order. Default is 3.
%
    if nargin < 1, dataset = 'random'; end

    if nargin < 2
        % parameters you can play with.
        opt.lambda = 1;        % regularization
        opt.loss = 'mlr';      % 'mlr' for Multinomial Logistic Regression
                               % (softmax) or 'l2svm' for L2 SVM.
        opt.dual = false;      % optimize dual problem
                               % (must be true to use kernels)
        opt.kernelfn = 'rbf';  % kernel to use (either rbf or poly)
        opt.gamma = 1e-2;      % Kernel parameter for RBF kernel.
        opt.order = 2;         % Kernel parameter for polynomial kernel.
    end
    
    % type the following into the matlab terminal to compile minFunc:
    % >> addpath ./minFunc/
    % >> mexAll
    addpath(genpath('./minFunc/'));
    addpath ./tinyclassifier/    
    addpath ./helpers
    
    if strcmp(dataset, 'random') % generate some random data.
        m = 150;  % number of data points per class
        n = 2;    % number of dimensions (features)
        K = 3;    % number of classes
        centers = 2*rand(K, n)-1;
        [X_train, y_train] = generateData(m, centers);
        [X_test, y_test] = generateData(2*m, centers);
        opt.display = true;    % plot decision boundary.
    else % load the given dataset.
        load(dataset);
        y_train = double(y_train);
        n = size(X_train, 2);
        ymin = min(y_train(:));
        y_train = y_train - ymin + 1;
        if ~exist('y_test', 'var')
            y_test = -ones(size(X_test, 1), 1); % dummy test labels.
        else
            y_test = double(y_test);
            y_test = y_test - ymin + 1;
        end
        K = max(y_train(:));
    end
%     Data = [X_train;X_test] ;
     
 %    [rows_train,cols_train] = size(X_train)
 %    [rows_test, cols_test] = size(X_test)
     
    % train and test classifier
%     [coeff,scores,latent] = pca(Data,'NumComponents',1000);
%    train_mat = scores(1:rows_train,:);
%    rows_train
%    size(train_mat)
%    test_mat = scores(rows_train+1:rows_train+rows_test,:);
%     rows_test
%    size(test_mat)
%    mdl = fitcknn(train_mat,y_train,'NumNeighbors',3,'NSMethod','exhaustive','Distance','minkowski','Standardize',1);
%    label = predict(mdl,Xnew);
%    size(label);
%     label

%reshape to imread



    % GIST Parameters:
% clear param
% param.imageSize = [256 256]; % set a normalized image size
% param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
% param.numberBlocks = 4;
% param.fc_prefilt = 4;
% 
% [Nimages,cols] = size(X_train);
% % Pre-allocate gist:
% Nfeatures = sum(param.orientationsPerScale)*param.numberBlocks^2;
% gist = zeros([Nimages Nfeatures]); 
% 
% % Load first image and compute gist:
%  img = reshape(X_train(1,:),32,32,3);
%  [gist(1, :), param] = LMgist(img, '', param); % first call
% % Loop:
% for i = 2:Nimages
%     img = reshape(X_train(i,:),32,32,3);
%     sprintf('%s %d','here train',i)
%   % img = X_train(i,:);
%    gist(i, :) = LMgist(img, '', param); % the next calls will be faster
% end
% 
% size(gist)
% save('./gist_train.mat','gist')
% [Nimages_test,cols] = size(X_test);
% Nfeatures = sum(param.orientationsPerScale)*param.numberBlocks^2;
% gist_test = zeros([Nimages_test Nfeatures]); 
% 
% % Load first image and compute gist:
% img_test = reshape(X_test(i,:),32,32,3);
% [gist_test(1, :), param] = LMgist(img_test, '', param); % first call
% % Loop:
% for i = 2:Nimages_test
%     img_test = reshape(X_test(i,:),32,32,3);
%     sprintf('%s here test %d','here test',i)
%   % img_test = X_test(i,:);
%   gist_test(i,:) = LMgist(img_test, '', param);
%   
% %   gist_test = [gist_test;gist_test(i,:)]; % the next calls will be faster
% end
% save('./gist_test.mat','gist_test')
% size(gist_test)
%     params = trainClassifier(gist(1:end,:), y_train(1:end), opt);
%     preds = predictClassifier(params, gist);
%     fprintf('Train Accuracy = %.2f%%\n', 100*mean(preds(:) == y_train(:)));
%     
%     preds = predictClassifier(params, gist_test);
%     fprintf('Test Accuracy = %.2f%%\n', 100*mean(preds(:) == y_test(:)));
%     
   
load('./gist_test.mat')
load('./gist_train.mat')
disp('model oaded')
size(gist)
size(gist_test)
model_pca = svmtrain(y_train,gist,'-c 100 -g 0.9 -t 2');
disp('train done')
[predict_label_L, accuracy_testpca1, dec_values_L] = svmpredict(y_test, gist_test, model_pca);
% write the data out to a file that can be read by Kaggle.
    writeLabels('my_labels.csv', predict_label_L);
    
    % plot the decision boundary or return if not plottable.
%     if n ~= 2 || ~isfield(opt, 'display') || ~opt.display, return; end;
%     
%     xmin = min(gist(:, 1));
%     xmax = max(gist(:, 1)); 100 , 0.06
%     ymin = min(gist(:, 2));
%     ymax = max(gist(:, 2));
%     
%     if K <= 6 % make pretty colors for scatter plot for up to 6 classes.
%         colors = [0 0 1; 1 0 0; 0 1 0; 0 1 1; 1 1 0; 1 0 1];
%         light_colors = colors + .8;
%         light_colors(light_colors > 1) = 1;
%         colormap(light_colors(1:K, :));
%     end
%     
%    % hold on;
%     plotBoundary(@(x) predictClassifier(params, x),...
%                  xmin, xmax, ymin, ymax);
%     if K <= 6, cy = colors(y_train,:); else cy = y_train; end;
%     scatter(gist(:, 1), gist(:, 2), [], cy, 'filled');
%     axis equal tight
%     hold off;
end

function [X, y] = generateData(m, centers)
% Generates some random guassian data.
%  m - number of data points to generate per class
%  centers - K x n matrix of cluster centers
% 
%  X - m*K x n design matrix
%  y - m*K x 1 labels
%
    [K, n] = size(centers);
    X = zeros(m*K, n);
    y = zeros(m*K, 1);
    for i = 1:K
        start_idx = (i-1)*m+1;
        end_idx = i*m;
        data = bsxfun(@plus, 0.2*randn(m, n), centers(i, :));
        X(start_idx:end_idx, :) = data;
        y(start_idx:end_idx) = i;
    end
end

function plotBoundary(predictClass, xmin, xmax, ymin, ymax)
% Plots decision boundary of this classifier.
%  predictClass - function that takes a set of data and predicts class
%                 label.
%  xmin, xmax, ymin, ymax - bounds to plot.
%
    if nargin < 4, ymin = xmin; ymax = xmax; end

    xrnge = linspace(xmin, xmax, 300);
    yrnge = linspace(ymin, ymax, 300);

    [xs, ys] = meshgrid(xrnge, yrnge);
    X = [xs(:), ys(:)];
    y = predictClass(X);
    y = reshape(y, size(xs));
    K = max(y(:));
    
    contourf(xs, ys, (y-1)./(K-1), K-1);
end
