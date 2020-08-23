classdef corrRegressionLayer < nnet.layer.RegressionLayer
    % Example custom regression layer with Negative-Pearson-Correlation-Coefficient (NPCC) loss.
        
    methods
        function layer = corrRegressionLayer(name)           
            % layer = maeRegressionLayer(name) creates a
            % NPCC regression layer and specifies the layer
            % name.
            
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "Negative Pearson Correlation Coefficient";
        end

        function loss = forwardLoss(~, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the NPCC loss between
            % the predictions Y and the training targets T.
            
            % Calculate NPCC.
                     
            [m,n,R,N] = size(Y);
            c = R*N;
            Y = reshape(Y,m,n,c);
            T = reshape(T,m,n,c);
                               
            T0 = T - mean(T,[1 2]);
            Y0 = Y - mean(Y,[1 2]);
            T0_norm = arrayfun(@(idx) norm(T0(:,:,idx),'fro'),1:c);
            Y0_norm = arrayfun(@(idx) norm(Y0(:,:,idx),'fro'),1:c);
            
            npcc = -sum(T0.*Y0,[1 2])./permute(T0_norm.*Y0_norm,[1 3 2]);
            
            loss = mean(npcc,'all');            
            
        end
        
        function dLdY = backwardLoss(~, Y, T)            % Backward propagate the derivative of the loss function.
            %
            % Inputs:
            %         layer - Output layer
            %         Y     �C Predictions made by network
            %         T     �C Training targets
            %
            % Output:
            %         dLdY  - Derivative of the loss with respect to the predictions Y        

            % Layer backward loss function goes here.
            
            [m,n,R,N] = size(Y);
            K = m*n;            
            c = R*N;
            
            Y = reshape(Y,m,n,c);
            T = reshape(T,m,n,c);
                               
            T0 = T - mean(T,[1 2]);
            Y0 = Y - mean(Y,[1 2]);
            T0_norm = arrayfun(@(idx) norm(T0(:,:,idx),'fro'),1:c);
            Y0_norm = arrayfun(@(idx) norm(Y0(:,:,idx),'fro'),1:c);
            
            U = sum(T0.*Y0,[1 2]);
            V = permute(T0_norm.*Y0_norm,[1 3 2]);
            dUdY = (1 - 1/K)*T0;
            dVdY = (1 - 1/K)*permute(T0_norm./Y0_norm,[1 3 2]).*Y0;
            
            dLdY = (U.*dVdY - V.*dUdY) ./ (N*R*V.^2);
            dLdY = reshape(dLdY,m,n,R,N);
                      
        end
    end
end