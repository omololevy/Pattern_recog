import numpy as np

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01):
        # Initialize the perceptron with random weights and bias
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(num_inputs)  # Initialize weights randomly
        self.bias = np.random.rand()  # Initialize bias randomly

    def activation_function(self, x):
        # Define the activation function (Step function)
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # Compute the output of the perceptron
        # Compute the weighted sum of inputs and apply activation function
        linear_combination = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(linear_combination)

    def train(self, training_inputs, labels, epochs):
        # Train the perceptron using the perceptron learning rule
        for epoch in range(epochs):
            # Iterate over each epoch
            for inputs, label in zip(training_inputs, labels):
                # Iterate over each training example and its label
                prediction = self.predict(inputs)
                # Make prediction using current weights
                error = label - prediction
                # Compute the error
                self.weights += self.learning_rate * error * inputs
                # Update weights based on the perceptron learning rule
                self.bias += self.learning_rate * error
                # Update bias based on the perceptron learning rule

# Example usage:
# Define training data (inputs and corresponding labels)
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input features
labels = np.array([0, 0, 0, 1])  # Corresponding target labels

# Create and train the perceptron
num_inputs = training_inputs.shape[1]
perceptron = Perceptron(num_inputs)
perceptron.train(training_inputs, labels, epochs=10)

# Test the trained perceptron
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for inputs in test_inputs:
    prediction = perceptron.predict(inputs)
    print(f"Inputs: {inputs}, Prediction: {prediction}")
