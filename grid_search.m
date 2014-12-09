% A grid search for parameter tunning that we wrote to find the optimal SVM parameters
lambda_grid = 0:0.1:1;
gamma_grid = 0:0.01:0.1;
acc = 0;
best_l = 0;
best_g = 0;
for l=1:length(lambda_grid)
    for g=1:length(gamma_grid)
        lam = lambda_grid(l)
        gam = gamma_grid(g)
        cv_acc = runClassifier('data.mat', struct('lambda', lambda_grid(l), 'loss', 'mlr', 'dual', true, 'kernelfn', 'rbf', 'gamma', gamma_grid(g)));
        if cv_acc>=acc
            acc = cv_acc;
            best_l = lambda_grid(l);
            best_g = gamma_grid(g);
        end 
    end 
end 


        