% Define training data (inputs and corresponding labels)
training_inputs = [0 0; 0 1; 1 0; 1 1]; % Input features
labels = [0; 0; 0; 1]; % Corresponding target labels

% Create and train the perceptron
perceptron = Perceptron(size(training_inputs, 2));
perceptron.train(training_inputs, labels, 10);

% Test the trained perceptron
test_inputs = [0 0; 0 1; 1 0; 1 1];
for i = 1:size(test_inputs, 1)
    inputs = test_inputs(i, :);
    prediction = perceptron.predict(inputs);
    disp(['Inputs: ' num2str(inputs) ', Prediction: ' num2str(prediction)]);
end
