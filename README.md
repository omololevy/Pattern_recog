# Explanation

## Python

The Perceptron class is defined with methods to initialize the perceptron.

They are:

* (__init__)
* make predictions (predict)
* and train the perceptron (train).

During initialization (__init__), the perceptron's weights and bias are initialized with random values.

The activation_function method defines the step function, which is used as the activation function for the perceptron.

The predict method computes the output of the perceptron by computing the weighted sum of inputs and applying the activation function.

The train method trains the perceptron using the perceptron learning rule. It iterates over the training examples and updates the weights and bias based on the error between the predicted and actual labels.

In the example usage, training data and labels are defined, and a perceptron is created and trained on the training data.

Finally, the trained perceptron is tested on some test inputs, and the predictions are printed.

## MatLab

**Perceptron Class Definition (`Perceptron.m`):**

```matlab
classdef Perceptron
    properties
        num_inputs
        learning_rate
        weights
        bias
    end
```

* The `Perceptron` class is defined with properties `num_inputs`, `learning_rate`, `weights`, and `bias`.

* These properties represent the number of input features, the learning rate, the weights associated with input features, and the bias term, respectively.

```matlab
    methods
        function obj = Perceptron(num_inputs, learning_rate)
            % Constructor method to initialize the perceptron
            if nargin < 2
                learning_rate = 0.01; % Default learning rate
            end
            obj.num_inputs = num_inputs;
            obj.learning_rate = learning_rate;
            obj.weights = rand(1, num_inputs); % Initialize weights randomly
            obj.bias = rand(); % Initialize bias randomly
        end
```

* The constructor method `Perceptron` initializes a perceptron object with the specified number of input features and learning rate.

* If the learning rate is not provided, it defaults to 0.01.

* It initializes the weights and bias with random values using the `rand` function.

```matlab
        function output = activation_function(obj, x)
            % Step function as the activation function
            output = x >= 0;
        end
```

* The `activation_function` method defines the step function as the activation function for the perceptron.

* It returns 1 if the input \(x\) is greater than or equal to 0, and 0 otherwise.

```matlab
        function prediction = predict(obj, inputs)
            % Method to compute the output of the perceptron
            % Compute the weighted sum of inputs and apply activation function
            linear_combination = dot(inputs, obj.weights) + obj.bias;
            prediction = obj.activation_function(linear_combination);
        end
```

* The `predict` method computes the output of the perceptron for a given input.

* It computes the weighted sum of inputs and applies the activation function defined earlier.

```matlab
        function train(obj, training_inputs, labels, epochs)
            % Method to train the perceptron using the perceptron learning rule
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
```

* The `train` method trains the perceptron using the perceptron learning rule.

* It iterates over a specified number of epochs and updates the weights and bias based on the perceptron learning rule.

* For each epoch, it iterates over each training example, computes the prediction, calculates the error, and updates the weights and bias accordingly.

**Example Usage (`perceptron_example.m`):**

```matlab
% Define training data (inputs and corresponding labels)
training_inputs = [0 0; 0 1; 1 0; 1 1]; % Input features
labels = [0; 0; 0; 1]; % Corresponding target labels
```

* This section defines the training data, including input features and corresponding target labels.

```matlab
% Create and train the perceptron
perceptron = Perceptron(size(training_inputs, 2));
perceptron.train(training_inputs, labels, 10);
```

* It creates an instance of the `Perceptron` class with the number of input features obtained from the size of the training inputs.
* It trains the perceptron on the training data for 10 epochs.

```matlab
% Test the trained perceptron
test_inputs = [0 0; 0 1; 1 0; 1 1];
for i = 1:size(test_inputs, 1)
    inputs = test_inputs(i, :);
    prediction = perceptron.predict(inputs);
    disp(['Inputs: ' num2str(inputs) ', Prediction: ' num2str(prediction)]);
end
```

* It tests the trained perceptron on some test inputs.
* It iterates over each test input, computes the prediction using the `predict` method, and displays the inputs along with the prediction.

This example demonstrates how to create, train, and use a perceptron in MATLAB.
