clear;
load data;
format long g;

X = wpbc(:,4:35);
[r c] = find(isnan(X));
X(r,:) = [];
wpbc(r,:) = [];

labels = wpbc(:,2);
positive_idx = find(labels==1);
negative_idx = find(labels==0);

np = size(positive_idx,1);
nn = size(negative_idx,1);

for v = 1:5
    
    rand_positive_idx = randi(np,6,1);
    rand_negative_idx = randi(nn,8,1);

    positive_values = positive_idx(rand_positive_idx);
    negative_values = negative_idx(rand_negative_idx);

    p = zeros(np,1);
    n = zeros(nn,1);

    p(rand_positive_idx) = 1;
    n(rand_negative_idx) = 1;



    train_negative_set_idx = negative_idx(n==0); 
    train_positive_set_idx = positive_idx(p==0); 
    train_set_idx = vertcat(train_negative_set_idx, train_positive_set_idx);

    validation_negative_set_idx = negative_idx(n==1); 
    validation_positive_set_idx = positive_idx(p==1); 
    validation_set_idx = vertcat(validation_negative_set_idx, validation_positive_set_idx);


    y = wpbc(:,3);
    
    [n p] = size(X);
    X = X - ones(n,1)*mean(X);
    X = X./sqrt(ones(n,1)*sum(X.^2));
    [n p] = size(y);
    y = y - ones(n,1)*mean(y);
    
    tX = X(train_set_idx,:);
    ty = y(train_set_idx);


    kfold = 4;
    indices = crossvalind('Kfold',ty,kfold);
    %fprintf('regression tree\n');
    sumsqe = zeros(kfold,1);
    for i = 1:kfold
        test = (indices == i); train = find(~test);
        test = find(test);
        if size(test,1) > 0
            tree(i)=rtree(tX(train,:),ty(train));
            yhat = tree(i).predict(tX(test,:));     
            sumsqe(i) = sum((yhat' - ty(test)).^2)/size(test,1);
    
        end
    end
    tree_mse(v) = mean(sumsqe);
    [~, idx] = min(sumsqe);
    best_tree = tree(idx);
    yhat = best_tree.predict(X(validation_set_idx,:));
    tree_validation_error(v) = sum((yhat' - y(validation_set_idx)).^2)/size(validation_set_idx,1);
    %fprintf('cart run(%d):  mse = %f, validation error = %f\n', v, tree_mse(v), tree_validation_error(v));
    
    %fprintf('lasso\n');
    sumsqe = zeros(kfold,1);

    for i = 1:kfold
        test = (indices == i); train = find(~test);
        test = find(test);
        if size(test,1) > 0
            lasso(i) = larslasso(tX(train,:),ty(train));
            yhat = lasso(i).predict(tX(test,:));
            sumsqe(i) = sum((yhat' - ty(test)).^2)/size(test,1);
        end
    end
    lasso_mse(v) = mean(sumsqe);

    [~, idx] = min(sumsqe);
    best_lasso = lasso(idx);
    yhat = best_lasso.predict(X(validation_set_idx,:));
    lasso_validation_error(v) = sum((yhat' - y(validation_set_idx)).^2)/size(validation_set_idx,1);

    %fprintf('lasso run(%d):  mse = %f, validation error = %f\n', v, lasso_mse(v), lasso_validation_error(v));

end

tree_mse'
lasso_mse'

tree_validation_error'
lasso_validation_error'
