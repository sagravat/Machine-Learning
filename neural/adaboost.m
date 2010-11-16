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

T = 10; % 10 weak learners
alpha = []; % weight of each weak learner 


net = {};
err = [];

num_groups = {};
num_groups{1} = [1:25];
num_groups{2} = [26:40];
num_groups{3} = [41:56];
num_groups{4} = [57:74];

train_limit = round(.90 * ylen);

valid_start_ind = train_limit + 1;
valid_end_ind = round(ylen*.95);
num_valid = valid_end_ind - valid_start_ind;

test_start_ind = valid_end_ind + 1;

num_test = ylen - test_start_ind;
feature_ind = [];

for i=1:7
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
        D = ones(num_test,1)/num_valid; % initial weight 
        tmp_net = {};
        valid_out = {};
        
        for g=1:length(num_groups)
            
            repeat = true;
            while repeat
                
                rand_gen = randi(40,1);
                r = randi(ylen,ylen,1);
                
                train_inp = X(r(1:train_limit),:);
                train_out = target(r(1:train_limit),:);
                
                valid_inp = X(r(valid_start_ind:valid_end_ind),:);
                valid_out{g} = target(r(valid_start_ind:valid_end_ind),:);
                tmp_net{g} = mlp(train_inp(:,num_groups{g}),train_out, rand_gen);
                
                tmp_err(t) = tmp_net{g}.mlptrain(train_inp(:,num_groups{g}),...
                     train_out,0.1,500);
                
                [cm, outputs] = tmp_net{g}.testmlp(valid_inp(:, num_groups{g}), valid_out{g});
                        

                perc_err(t,g) = trace(cm)/sum(sum(cm))*100;
                repeat = false;
                if perc_err(t,g) < 53
                    repeat = true;
                end
            end
            
            [max_corr, ind] = max(perc_err(t,:));
            
            
            if( max(perc_err(t,:)) > max_perc)
                outs = outputs;
                error = sum(D.*(1-max_corr/100)*num_valid)/100;
                alpha(t) = 1/2 * log((1-error)/error);
                net{t} = tmp_net{ind};
                max_perc =  max(perc_err(t,:));
                feature_ind(t) = ind;
                valids = valid_out{ind};
                valids(find(valids(:,1)==0))=-1;
                valids(:,2) = [];
                %fprintf('i = %d, t=%d, percent correct = %f, rand_gen = %d\n', i, ...
                %    t, max_corr, rand_gen);
            end
        end
        
        
        D = D.* exp(-alpha(t).*valids.* outs);
        D = D./sum(D);
        
        
    end
    
    test_start_ind = valid_end_ind + 1;
    test_inp = X(r(test_start_ind:end),:);
    test_out = target(r(test_start_ind:end),:);
    
    finalLabel = zeros(size(test_out,1),1);
    
    
    for t=1:length(alpha)
        [cm, outputs] = net{t}.testmlp(test_inp(:,num_groups{feature_ind(t)}), test_out);
        finalLabel = finalLabel + alpha(t) * outputs;
        fprintf('features = %s, hidden nodes = %d\n', ...
            mat2str(num_groups{feature_ind(t)}), net{t}.num_hidden);
    end
    
    s = sign(finalLabel);
    p = test_out;
    p(p==0)=-1;
    p(:,2)=[];
    corr = length(find(s==p))/length(s);
    fprintf('i = %d, corr= %f\n', i, corr);
    
end


