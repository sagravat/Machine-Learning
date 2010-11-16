
clear all;

load pathology;

y = y - 1;
indices = 1: length(y);

X = X(indices,:);
y = y(indices);

X = repmat(mean(X),length(X),1)-X;
m = max(X);
X = X./repmat(m,length(X),1);


ylen = length(indices);

target = zeros(ylen,2);

T = 5; % k = 5 for cross validation

net = {};
err = [];

num_groups = {};
num_groups{1} = [1:74];

train_limit = round(.90 * ylen);

valid_start_ind = train_limit + 1;
valid_end_ind = round(ylen*.95);
num_valid = valid_end_ind - valid_start_ind;

test_start_ind = valid_end_ind + 1;

num_test = ylen - test_start_ind;
feature_ind = [];
corr = [];
valid_test = [];
stats = {};
for i=1:7
    for z=1:30
        target(find(y == i)) = 1;
        perc_err = zeros(10,4);
        for k=1:7
            if (k ~= i)
                target(find(y == k),2) = -1;
            end
        end
        
        for t = 1:T
            err(t) = inf;
            max_perc = 0;
            tmp_net = {};
            valid_out = {};
            
            rand_gen = randi(40,1);
            r = randi(ylen,ylen,1);
            
            train_inp = X(r(1:train_limit),:);
            train_out = target(r(1:train_limit),:);
            
            valid_inp = X(r(valid_start_ind:valid_end_ind),:);
            valid_out = target(r(valid_start_ind:valid_end_ind),:);
            tmp_net = mlp(train_inp,train_out, rand_gen);
            
            tmp_err(t) = tmp_net.mlptrain(train_inp,...
                train_out,0.1,1000);            
            
            [cm, outputs] = tmp_net.testmlp(valid_inp(:, num_groups{1}), valid_out);
            max_corr = trace(cm)/sum(sum(cm))*100;
            repeat = false;
            
            if( max_corr > max_perc)
                net{t} = tmp_net;
                max_perc =  max_corr;
                valid_test(z) = max_perc;
            end
            
        end
        fprintf('[valid] i = %d, corr= %f\n', i, max_perc);
        
        test_start_ind = valid_end_ind + 1;
        test_inp = X(r(test_start_ind:end),:);
        test_out = target(r(test_start_ind:end),:);
        
        [cm, outputs] = net{t}.testmlp(test_inp(:,num_groups{1}), test_out);
        
        corr(z) = trace(cm)/sum(sum(cm))*100;
        
        fprintf('[test] i = %d, corr= %f\n\n', i, corr(z));
        
    end
    stats{i} = corr;
    
end


