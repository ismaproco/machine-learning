function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

      temp = theta;
      
      % calculate theta 0
      ji = 0;
      for i = 1:m
        ji = ji + ( temp(1) + temp(2) * X(i, 2) ) - y(i);
      end
      
      theta(1) = temp(1) - alpha * ( ji )/m;

      %calculate theta 1

      ji = 0;
      for i = 1:m
        basic = ( ( temp(1) + temp(2) * X(i, 2) ) - y(i) );
        ji = ji + ( basic * X(i, 2) ) ;
      end
      
      theta(2) = temp(2) - alpha * ( ji )/m;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
