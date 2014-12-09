% a voting mechanism to combine multiple ensembles

function [output] = get_majority(mtx)

    N_samples = size(mtx,1);
    
    output = mode(mtx,2);
    
    for i=1:N_samples
        mode_pred = output(i);
        predictions = mtx(i,:);

        num_mode = sum(predictions==mode_pred);
        if (num_mode==1)
            output(i) = mtx(i,1); % heuristic: taking the first classifier (strongest) prediction
        end 
    end       
end