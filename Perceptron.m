classdef Perceptron
    properties
        num_inputs
        learning_rate
        weights
        bias
    end
    
    methods
        function obj = Perceptron(num_inputs, learning_rate)
            if nargin < 2
                learning_rate = 0.01; % Default learning rate
            end
            obj.num_inputs = num_inputs;
            obj.learning_rate = learning_rate;
            obj.weights = rand(1, num_inputs);
            obj.bias = rand();
        end
        
        function output = activation_function(obj, x)
            % Step function as the activation function
            output = x >= 0;
        end
        
        function prediction = predict(obj, inputs)
            % Compute the output of the perceptron
            % Compute the weighted sum of inputs and apply activation function
            linear_combination = dot(inputs, obj.weights) + obj.bias;
            prediction = obj.activation_function(linear_combination);
        end
        
        function train(obj, training_inputs, labels, epochs)
            % Train the perceptron using the perceptron learning rule
            for epoch = 1:epochs
                % Iterate over each epoch
                for i = 1:size(training_inputs, 1)
                    % Iterate over each training example and its label
                    inputs = training_inputs(i, :);
                    label = labels(i);
                    prediction = obj.predict(inputs);
                    error = label - prediction;
                    obj.weights = obj.weights + obj.learning_rate * error * inputs;
                    obj.bias = obj.bias + obj.learning_rate * error;
                end
            end
        end
    end
end
