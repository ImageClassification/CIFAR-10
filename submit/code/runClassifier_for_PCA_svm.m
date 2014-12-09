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
     Data = [X_train;X_test] ;
     
     [rows_train,cols_train] = size(X_train);
     [rows_test, cols_test] = size(X_test);
     
    % train and test classifier
    [coeff,scores,latent] = pca(Data,'NumComponents',512);
   train_mat = scores(1:rows_train,:);
   rows_train
   size(train_mat)
   test_mat = scores(rows_train+1:rows_train+rows_test,:);
    rows_test
   size(test_mat)
   
    
    params = trainClassifier(train_mat(1:end,:), y_train(1:end), opt);
    preds = predictClassifier(params, train_mat);
    fprintf('Train Accuracy = %.2f%%\n', 100*mean(preds(:) == y_train(:)));
    
    preds = predictClassifier(params, test_mat);
    fprintf('Test Accuracy = %.2f%%\n', 100*mean(preds(:) == y_test(:)));
    
    % write the data out to a file that can be read by Kaggle.
    writeLabels('my_labels.csv', preds);
    
    % plot the decision boundary or return if not plottable.
    if n ~= 2 || ~isfield(opt, 'display') || ~opt.display, return; end;
    
    xmin = min(train_mat(:, 1));
    xmax = max(train_mat(:, 1));
    ymin = min(train_mat(:, 2));
    ymax = max(train_mat(:, 2));
    
    if K <= 6 % make pretty colors for scatter plot for up to 6 classes.
        colors = [0 0 1; 1 0 0; 0 1 0; 0 1 1; 1 1 0; 1 0 1];
        light_colors = colors + .8;
        light_colors(light_colors > 1) = 1;
        colormap(light_colors(1:K, :));
    end
    
    hold on;
    plotBoundary(@(x) predictClassifier(params, x),...
                 xmin, xmax, ymin, ymax);
    if K <= 6, cy = colors(y_train,:); else cy = y_train; end;
    scatter(train_mat(:, 1), train_mat(:, 2), [], cy, 'filled');
    axis equal tight
    hold off;
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
