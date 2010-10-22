classdef larslasso
    properties(SetAccess='protected',GetAccess='public')
        beta;
    end
    
    methods
        

        function l = larslasso(X,y)

            [n p] = size(X);

            num_features = min(n-1,p);
            yhat = zeros(n,1);
            active = [];
            inactive = 1:p;
            beta = zeros(p, 1);
            cov_X = X'*X;
            i = 0;
            drop_variable = false;
          
            while i < num_features
                

                if size(inactive,1) == 0
                    break;
                end
                % c hat is the vector of current correlations 
                c_hat = X'*(y - yhat);

                % covariate with greatest absolute correlation
                [C_hat, j] = max(abs(c_hat(inactive)));    
                j = inactive(j);
                if ~drop_variable
                    active = [active j];
                    inactive(inactive==j) = [];
                    i = i + 1;
                end

                drop_variable = false;
                % find sign for all active correlations
                s = sign(c_hat(active));

                % g_active 
                num_active = size(active',1);
                g_active = (cov_X(active, active).* (s*ones(1,num_active))' .* (s*ones(1,num_active)))^-1 * ones(num_active,1) ;
                a_a = 1/sqrt(ones(1,num_active) * (g_active ));

                
                % weights for each variable
                w_a = a_a * g_active .* s;
                % u_a = equiangular unit vector
                u_a = X(:,active)*w_a;

               
                % residual of greatest correlation and inactive variables with
                % ratio to correlation of equiangular unit vector    
                if num_active == num_features 
                    % solve linear regression since all variables are
                    % active
                    gamma_hat = C_hat/a_a;
                else
                    % correlation for variables vs. equiangular unit vecotr
                    a = X'*u_a;

                    r1 = (C_hat-c_hat(inactive))./(a_a - a(inactive));
                    r2 = (C_hat+c_hat(inactive))./(a_a + a(inactive));
                    residual_ratio = [r1 ; r2];
                    gamma_hat = min([residual_ratio(residual_ratio>0); C_hat/a_a]);

                end

                % modification to LARS starts here
                gamma = -beta(active)./w_a;
 
                new_gamma = min([gamma(gamma > 0)' gamma_hat]);    
                
                if new_gamma < gamma_hat
                    gamma_hat = new_gamma;          
                    drop_variable = true;        
                end

                beta(active) = beta(active) + gamma_hat * w_a;       
                yhat = yhat + gamma_hat*u_a;

                if drop_variable == true
                    idx = find(gamma == new_gamma);
                    inactive = [inactive active(idx)];        
                    active(idx) = [];        
                    i = i - 1;
                end
            end
            
            
            l.beta = beta;
        end
        
        function outcomes = predict(self, X)
            outcomes = [];
            
            for i=1:size(X,1)
               outcomes(i) = X(i,:)*self.beta(:); 
            end     
        end
    end
end