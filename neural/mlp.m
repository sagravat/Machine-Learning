classdef mlp < handle
    
    properties
        num_inputs = 0;
        num_outputs = 0;
        num_rows = 0;
        num_hidden = 0;
        
        beta = 1;
        momentum = 0.9;
        weights1 = [];
        weights2 = [];
        
        hidden = [];
        
    end 
    
    methods
        
        function obj = mlp(inputs, targets, num_hidden)
            obj.num_inputs = size(inputs,2);
            obj.num_outputs = size(targets,2);
            obj.num_rows = length(inputs);
            obj.num_hidden = num_hidden;
            obj.weights1 = (rand(obj.num_inputs+1,num_hidden) - .5)*2/sqrt(obj.num_inputs);
            obj.weights2 = (rand(num_hidden+1,obj.num_outputs) - .5)*2/sqrt(num_hidden);
            
        end
        
        function new_val_error = earlystopping(obj,inputs, targets, valid, validtargets, eta, niterations)
           valid = [valid -ones(size(valid,1),1)];
           
           old_val_error1 = 100002;
           old_val_error2 = 100001;
           new_val_error  = 100000;
           
           count = 0;
           
           while old_val_error1 - new_val_error > 1e-3 || old_val_error2 - old_val_error1 > 1e-3
              count = count + 1;
              obj.mlptrain(inputs, targets, eta, niterations);
              old_val_error2 = old_val_error1;
              old_val_error1 = new_val_error;
              validout = obj.mlpfwd(valid);
              new_val_error = 0.5*sum(sum(validtargets - validout).^2);
              
              %fprintf('stopped: %d, %d, %d\n', new_val_error, old_val_error1, old_val_error2);
        
           end
           
        end
        
        function error = mlptrain(obj,inputs, targets, eta, niterations)
            inputs = [inputs -ones(obj.num_rows,1)];
            change = length(obj.num_rows);
            updatew1 = zeros(size(obj.weights1));
            updatew2 = zeros(size(obj.weights2));
            error = 1e5;
             for n=1:niterations
               outputs = obj.forward(inputs);
               error = 0.5*sum(sum((targets-outputs).^2));
               %if (mod(n,100)==0)
               %    fprintf('iteration: %d, Error: %f\n', n, error);
               %end
               deltao = (targets-outputs).*outputs.*(1.0-outputs);

               deltah = obj.hidden.*(1.0-obj.hidden).* (deltao*obj.weights2');

               updatew1 = eta .* (inputs' * deltah(:,1:end-1)) + obj.momentum .* updatew1;
               updatew2 = eta .* (obj.hidden' * deltao) + obj.momentum .* updatew2;
               obj.weights1 = obj.weights1 + updatew1;
               obj.weights2 = obj.weights2 + updatew2;
               % Randomise order of inputs
              
               r = randi(change,change,1);
               inputs = inputs(r,:);
               targets = targets(r,:);
%               fprintf('error = %f\n', error);

            end
            
            


            
        end
        
        function outputs = forward(obj, inputs)
            obj.hidden = inputs*obj.weights1;
            obj.hidden = 1.0./(1.0+exp(-obj.beta.*obj.hidden));
            obj.hidden = [obj.hidden -ones(size(inputs,1), 1)];
            outputs = obj.hidden * obj.weights2;    
            %outputs = 1.0./(1.0+exp(-obj.beta.*outputs));
            

            outputs = softmax(outputs);

            %normalisers = sum(exp(outputs),2)'.*ones(1,size(outputs,1));
            %outputs = (exp(outputs)./repmat(normalisers,obj.num_outputs,1)');

        end
        
        function [cm, outputs] = testmlp(obj, inputs, targets)
            inputs = [inputs -ones(size(inputs,1),1)];
            outputs = obj.forward(inputs);

            nclasses = size(targets,2);

            [~,outputs] = max(outputs');
            targets(targets(:,1)==0)=2;
            targets(:,2) = [];
            targets=targets';
            
            cm = zeros(nclasses, nclasses);
            for i=1:nclasses
                for j=1:nclasses
                    o = zeros(size(outputs,2),1)';
                    ind1 = find(outputs==i);
                    o(ind1) = 1;
                    m = zeros(size(targets,2),1)';
                    ind2 = find(targets==j);
                    m(ind2) = 1;
                    
                    cm(i,j) = sum(o .* m);
                end
            end
            outputs = outputs';
            outputs(outputs==2)=-1;
            %fprintf('percent correct: %f\n', trace(cm)/sum(sum(cm))*100);


        end
        
    end
    
end
