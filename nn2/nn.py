import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self, layers, activation_fun='sigmoid'):
        """
        layers: list of integers, number of neurons in each layer (e.g. [2, 3, 1] for 2 input neurons, 3 neurons in the hidden layer, and 1 output neuron)
        This defines the necessary dimensions for the weight matrices and bias vectors.      
    """
        self.layers = layers
        self.weights = []
        self.biases = []
        self.activation_fun = activation_fun
        self.weights, self.biases = self.generate_random_weights(activation_fun=activation_fun)
        self.layer_values = [0] * (len(self.layers) - 1)
        self.best_weights = None
        self.best_biases = None
        self.best_mse = np.inf
        self.model_age = 0

    def generate_random_weights(self, activation_fun='sigmoid'):
        weights, biases = [], []
        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            fan_out = self.layers[i+1]
            if activation_fun == 'sigmoid':
                limit = np.sqrt(6 / (fan_in + fan_out))  # Xavier for sigmoid
            if activation_fun == 'relu':
                limit = np.sqrt(2 / fan_in)  # He for ReLU
            weight_matrix = np.random.uniform(-limit, limit, (fan_in, fan_out))
            bias_vector = np.zeros((1, fan_out))  # Common to initialize biases to 0
            weights.append(weight_matrix)
            biases.append(bias_vector)
        return weights, biases
    
    def activation(self, x):
        if self.activation_fun == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        if self.activation_fun == 'relu':
            return np.maximum(0, x)
        
    def activation_derivative(self, x):
        if self.activation_fun == 'sigmoid':
            return self.activation(x) * (1 - self.activation(x))
        if self.activation_fun == 'relu':
            return (x > 0).astype(float)
    
    def forward(self, x, best_model=False):
        if best_model:
            self.weights = self.best_weights
            self.biases = self.best_biases
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(a, w) + b
            self.layer_values[i] = z  # Store activations for backprop
            a = self.activation(z) if i < len(self.weights) - 1 else z        
        return a
    
    def backward(self, x, y, y_pred, learning_rate):
        error = y_pred - y 
        delta = error

        for i in reversed(range(len(self.layer_values))):
            if i < len(self.layer_values) - 1:  # For Hidden layers
                delta = np.dot(delta, self.weights[i + 1].T) * self.activation_derivative(self.layer_values[i])

            # Weights and biases update
            input = x if i == 0 else self.layer_values[i - 1]
            self.weights[i] -= learning_rate * np.clip(np.dot(input.T, delta), -2, 2)
            self.biases[i] -= learning_rate * np.mean(delta, axis=0, keepdims=True)


    def train(self, x, y, learning_rate, epochs, mini_batch=False, batch_size=32, stop_condition=0.5, report_interval=1000):
        
        num_samples = x.shape[0]
        history = []
        pred_history = []
        weight_history = []

        start_mse = np.mean(np.square(y - self.forward(x)))

        print(f"Starting MSE: {start_mse:.2f}")
        if mini_batch:
            for epoch in range(epochs):
                indices = np.random.permutation(num_samples)  # Shuffle dataset
                for i in range(0, num_samples, batch_size):
                    batch_x = x[indices[i:i + batch_size]]
                    batch_y = y[indices[i:i + batch_size]]

                    y_pred = self.forward(batch_x)
                    self.backward(batch_x, batch_y, y_pred, learning_rate)

                mse = np.mean(np.square(y - self.forward(x)))
                history.append(mse)
                weight_history.append(self.weights[0][0][0])

                if epoch % report_interval == 0:
                    print(f"Epoch {self.model_age + epoch}, MSE: {mse:.2f}")
                    # pred_history.append(self.forward(x))

                if mse < stop_condition or not np.isfinite(mse):
                    break

                if mse < self.best_mse:
                    self.best_mse = mse
                    self.best_weights = self.weights
                    self.best_biases = self.biases

        else:
            for epoch in range(epochs):
                y_pred = self.forward(x)
                mse = np.mean(np.square(y - y_pred))
                history.append(mse)
                if mse < stop_condition or not np.isfinite(mse):
                    break
                self.backward(x, y, y_pred, learning_rate)
                if epoch % report_interval == 0:
                    print(f"Epoch {self.model_age + epoch}, MSE: {mse:.2f}")
                    # pred_history.append(y_pred)
                weight_history.append(self.weights[0][0][0])

                if mse < self.best_mse:
                    self.best_mse = mse
                    self.best_weights = self.weights
                    self.best_biases = self.biases

        self.model_age += epoch + 1
        
        print(f"Final MSE: {mse} after {self.model_age} epochs")

        # self.animate_training(x, y, pred_history)
        return history, weight_history
    
    def animate_training(self, x, y, pred_history):
        fig, ax = plt.subplots()
        true_scatter = ax.scatter(x, y, color='blue', label='True values')
        pred_scatter = ax.scatter(x, pred_history[0], color='red', label='Predicted values')
        ax.legend()

        for pred in pred_history:
            pred_scatter.set_offsets(np.c_[x, pred])
            
            plt.draw()
            plt.pause(0.1)

        plt.show()
        
    def predict(self, x):
        y_pred = self.forward(x, best_model=True)
        return y_pred

    def set_weights(self, weights, biases):
        self.weights = weights
        self.biases = biases
    
    def print_weights(self):
        print("Weights: ", self.weights)
        print("Biases: ", self.biases)

    def print_layer_values(self):
        print("Layer values: ", self.layer_values)
    
    def get_layer_values(self):
        return self.layer_values
