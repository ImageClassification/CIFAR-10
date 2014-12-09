load('../data.mat');
%%
% loads data
load('../data.mat');
X_heldout = X_train(:,1:2400)';
X_cv = X_train(:,2401:end)';
Y_heldout = y_train(1:2400);
Y_cv = y_train(2401:end);


tic; disp('fitting 1000 weak learners with tree');
% each row of X contains 1 observation 
ens = fitensemble(X_train,y_train,'AdaboostM2',300,'Tree');

toc; disp('done');

%% create cross validated ensemble method to test a few variables
% code from: http://www.mathworks.com/help/stats/classification-trees-and-regression-trees.html
% examine the error as the trees rise, is the best leaf number the same for 
% increasing number of trees?
tic; disp('fitting 50 weak learners with tree');
cv = fitensemble(X_train,y_train,'AdaboostM2',50,'Tree',...
                 'kfold',5);
toc; disp('done');
%% visualize
figure;
plot(kfoldLoss(cv,'mode','cumulative'),'r.');
xlabel('Number of trees');
ylabel('Classification error');
legend('Test','Cross-validation','Location','NE');

%%
tic; disp('fitting 500 weak learners with tree');
cv100 = fitensemble(X_train,y_train,'AdaboostM2',500,'Tree',...
                 'kfold',5);
toc; disp('done');         

%%
figure;
plot(kfoldLoss(cv1500,'mode','cumulative'),'r.');
xlabel('Number of trees');
ylabel('Classification error');
legend('CIFAR-10','Cross-validation','Location','NE');

%%
tic; disp('fitting 1500 weak learners with tree');
cv1500 = fitensemble(X_train,y_train,'AdaboostM2',1500,'Tree',...
                 'kfold',5);
toc; disp('done');         

%%
figure;
plot(kfoldLoss(cv1500,'mode','cumulative'),'r.');
xlabel('Number of trees');
ylabel('Classification error');
legend('CIFAR-10','Cross-validation','Location','NE');

%%
tic; disp('fitting 5000 weak learners with tree');
cv5000 = fitensemble(X_train,y_train,'AdaboostM2',5000,'Tree',...
                 'kfold',5);
toc; disp('done');         

%%
figure;
plot(kfoldLoss(cv5000,'mode','cumulative'),'r.');
xlabel('Number of trees');
ylabel('Classification error');
legend('CIFAR-10','Cross-validation','Location','NE');

%% make predictions using the learned ensemble

preds = predict(cv5000,X_test);

%% 
tic;
c5000 = fitensemble(X_train,y_train,'AdaboostM2',5000,'tree');
toc;
%%
preds = predict(c5000,X_test);
