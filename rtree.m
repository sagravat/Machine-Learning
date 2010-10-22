classdef rtree < handle
    properties(SetAccess='protected',GetAccess='public')
        X = [];
        y = [];
        count = 1;
        feature_tracker = [];
        node = struct();
    end
   
    methods
        function obj  = rtree(X,y)
            obj.X = X;
            obj.y = y;
            [n p] = size(X);
           
            obj.feature_tracker = zeros(p,1);
            [threshold, feature_col] = find_split_point(obj, X, y);
            A=obj.X(:,feature_col);
           
           
            obj.node.feature = feature_col;
            obj.node.threshold = threshold;
            obj.node.prediction = NaN;
           
            obj.node.left = struct();
            obj.node.right = struct();
           
            LT = find(A<=threshold-1e-6);
            GT = find(A>threshold);
            tic;
            obj.node.left  = obj.create_tree(obj.node.left, LT);
            obj.node.right = obj.create_tree(obj.node.right, GT);
            toc;

        end % constructor

        function node = create_tree(obj, node, indices)
            [threshold, feature_col] = obj.find_split_point(obj.X(indices, :), obj.y(indices));
            if feature_col == 0
                node.feature = NaN;
                node.threshold = NaN;
                if isnan(mean(obj.y(indices)))
                    node.prediction = mean(obj.y);
                else
                    node.prediction = mean(obj.y(indices));
                end
                return
            end
            A=obj.X(:,feature_col);
            LT = find(A<=threshold-1e-6);
            GT = find(A>threshold);
            node.feature = feature_col;
            node.threshold = threshold;
            node.prediction = NaN;
            node.left  = struct();
            node.right = struct();
            node.left = obj.create_tree(node.left, LT);
            node.right = obj.create_tree(node.right, GT);

        end
       

        function [threshold, feature_col] = find_split_point(obj, X, y)
            [n, p] = size(X);

           
            split_points = zeros(p,1);
            min_sum = 1e8*ones(p,1);
            threshold = 0;
            feature_col = 0;
            %i = 1;
            if (size(find(obj.feature_tracker == 0),1) == 0)               
                return;
            end
            
            if n < 5
                return;
            end
            
            for x = 1:p
                if obj.feature_tracker(x) == 1
                    min_sum(x) = 1e8;
                else
                   
                    values = unique(X(:,x));
                    values(values==99999) = [];
                    A = X(:,x);
                    total_sum = 1e8*ones(size(values,1),1);
                    for i=1:size(values,1)
                        s = values(i);
                        if s ~= 99999
                            r1_indices = find(A<=s);
                            r2_indices = find(A>s);
                            c1 = mean(y(r1_indices));
                            c2 = mean(y(r2_indices));

                            S1 = (y(r1_indices) - c1).^2;
                            S2 = (y(r2_indices) - c2).^2;

                            total_sum(i) = sum(S1) + sum(S2);
                        end
                    end
                    [val, row] = min(total_sum);
                    split_points(x) = row;
                    min_sum(x) = val;
                end
            end
            [~, feature_col] = min(min_sum);           
           
            row = split_points(feature_col);
            threshold = X(row,feature_col);
            A = obj.X(:, feature_col);
            indices = A == threshold;
            obj.X(indices, feature_col) = 99999;
            num_marked = find(obj.X(:,feature_col)==99999);
            if size(num_marked,1) > 1
                obj.feature_tracker(feature_col) = 1;
            end
            obj.count = obj.count + 1;
        end
       
        function y = predict(obj, test)
            y = [];
            for i=1:size(test,1)
                prediction = NaN; 
                node = obj.node;
                while isnan(prediction)
                    feature = node.feature;
                    thresh  = node.threshold;
                    prediction = node.prediction;
                    if ~isnan(prediction)
                        break;
                    end
                    if test(i,feature) <= thresh
                        node = node.left;
                    else
                        node = node.right;
                    end
 
                end
                y(i) = prediction;    
            end
            
        end
    end % methods block
   
end


