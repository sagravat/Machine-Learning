alpha=.05;
sigma =2;

for i=1:7
    
    mu = mean(stats{i});
    cutoff1 = norminv(alpha, mu, sigma);
    cutoff2 = norminv(1-alpha, mu, sigma);    
    
    fprintf('variable %d vs all, 95%% confidence interval = %f - %f\n',...
        i, cutoff1, cutoff2);
    
end

fprintf('\n');
