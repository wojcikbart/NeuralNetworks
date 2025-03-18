import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self, X, y, layers,
                  activation_fun='sigmoid', output_activation='linear',
                  loss_fun='mse', regularization=None, reg_lambda=0.001,
                  momentum=0.9):
        """
        layers: list of integers, number of neurons in each layer (e.g. [2, 3, 1] for 2 input neurons, 3 neurons in the hidden layer, and 1 output neuron)
        This defines the necessary dimensions for the weight matrices and bias vectors.      
    """
        
        self.X = X
        self.y = y
        self.layers = layers
        self.weights = []
        self.biases = []
        self.activation = ActivationFunction(activation_fun)
        self.output_activation = ActivationFunction(output_activation)
        self.loss_fun = loss_fun
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.weights, self.biases = self.generate_random_weights(activation_fun=activation_fun)
        self.layer_values = [0] * (len(self.layers) - 1)
        self.best_weights = None
        self.best_biases = None
        self.best_mse = np.inf
        self.model_age = 0

        # momentum
        self.momentum = momentum
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.v_biases = [np.zeros_like(b) for b in self.biases]

    def generate_random_weights(self, activation_fun='sigmoid'):
        weights, biases = [], []
        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            fan_out = self.layers[i+1]
            if self.activation.name == 'sigmoid':
                limit = np.sqrt(6 / (fan_in + fan_out))  # Xavier for sigmoid
            if self.activation.name == 'relu':
                limit = np.sqrt(2 / fan_in)  # He for ReLU
            weight_matrix = np.random.uniform(-limit, limit, (fan_in, fan_out))
            bias_vector = np.zeros((1, fan_out))  # Common to initialize biases to 0
            weights.append(weight_matrix)
            biases.append(bias_vector)
        return weights, biases

    def loss(self, y_true, y_pred, include_regularization=True):
        if self.loss_fun == 'mse':
            loss = np.mean(np.square(y_true - y_pred))
        if self.loss_fun == 'mae':
            loss = np.mean(np.abs(y_true - y_pred))

        regularization = 0

        if self.regularization is not None and include_regularization:
            
            if self.regularization == 'l1':
                for w in self.weights:
                    regularization += np.sum(np.abs(w))
            if self.regularization == 'l2':
                for w in self.weights:
                    regularization += np.sum(np.square(w))

        return loss + self.reg_lambda * regularization
    
    def loss_derivative(self, y_true, y_pred):
        if self.loss_fun == 'mse':
            return y_pred - y_true
        if self.loss_fun == 'mae':
            return np.sign(y_pred - y_true)
    
    def forward(self, x, store_values=False, best_weights=False):
        if best_weights:
            weights = self.best_weights
            biases = self.best_biases
        else:
            weights = self.weights
            biases = self.biases

        a = x

        for i, (w, b) in enumerate(zip(weights, biases)):
            z = np.dot(a, w) + b
            if store_values:
                self.layer_values[i] = z
            a = self.activation.activate(z) if i < len(weights) - 1 else self.output_activation.activate(z)
        return a
    
    def backward(self, x, y, y_pred, learning_rate, grad_threshold=5):
        error = self.loss_derivative(y, y_pred) * self.output_activation.derivative(y_pred)
        delta = error

        for i in reversed(range(len(self.layer_values))):
            if i < len(self.layer_values) - 1:  # For Hidden layers
                delta = np.dot(delta, self.weights[i + 1].T) * self.activation.derivative(self.layer_values[i])

            # Weights and biases update
            input = x if i == 0 else self.layer_values[i - 1]
            gradient = np.dot(input.T, delta)
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > grad_threshold:
                gradient = grad_threshold * gradient / grad_norm

            self.v_weights[i] = self.momentum * self.v_weights[i] + learning_rate * np.clip(gradient, -grad_threshold, grad_threshold)
            self.v_biases[i] = self.momentum * self.v_biases[i] + learning_rate * np.mean(delta, axis=0, keepdims=True)

            
            self.weights[i] -= self.v_weights[i]
            self.biases[i] -= self.v_biases[i]

    def train(self, learning_rate, epochs, validation_data=None,
              mini_batch=False, batch_size=32,
              stop_condition=0.5, patience=1000, report_interval=1000):
        
        x = self.X
        y = self.y
        num_samples = x.shape[0]
        history = []
        wait = 0

        if self.best_weights is not None:
            self.weights = self.best_weights
            self.biases = self.best_biases
            
        start_mse = self.loss(y, self.forward(x))

        print(f"Starting MSE: {start_mse:.2f}")
        for epoch in range(epochs):
            if mini_batch:
                indices = np.random.permutation(num_samples)
                for i in range(0, num_samples, batch_size):
                    batch_x = x[indices[i:i + batch_size]]
                    batch_y = y[indices[i:i + batch_size]]     
                    y_pred = self.forward(batch_x, store_values=True)
                    self.backward(batch_x, batch_y, y_pred, learning_rate)
            else:
                y_pred = self.forward(x, store_values=True)
                self.backward(x, y, y_pred, learning_rate)

            current_train_loss = self.loss(y, self.forward(x))
            history.append(current_train_loss)

            if epoch > 0 and epoch % (patience // 2) == 0:
                if len(history) >= patience//2 and abs(history[-patience//2] - history[-1]) < 1e-4 and current_train_loss > 100:
                    learning_rate *= 0.5
                    print(f"Epoch {self.model_age + epoch}: Reducing learning rate to {learning_rate}")

            if wait > patience // 2:
                noise_scale = 0.01 * np.std(self.weights[0])
                for i in range(len(self.weights)):
                    self.weights[i] += np.random.normal(0, noise_scale, self.weights[i].shape)
                print("Added noise to enable escape from local minimum or plateau")
                wait = 0

            if validation_data is not None:
                x_val, y_val = validation_data
                val_loss = self.loss(y_val, self.forward(x_val))
                benchmark_loss = val_loss
            else:
                benchmark_loss = current_train_loss

            if benchmark_loss < self.best_mse:
                self.best_mse = benchmark_loss
                self.best_weights = self.weights
                self.best_biases = self.biases
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {self.model_age + epoch}")
                    break
            
            if benchmark_loss < stop_condition:
                print(f"Stopping condition met at epoch {self.model_age + epoch}")
                break

            if epoch % report_interval == 0:
                if validation_data is not None:
                    print(f"Epoch {self.model_age + epoch}, Train MSE: {current_train_loss:.2f}, Val MSE: {val_loss:.2f}")
                else:
                    print(f"Epoch {self.model_age + epoch}, Train MSE: {current_train_loss:.2f}")

        self.model_age += epoch + 1
        print(f"Training complete. Final loss: {self.best_mse:.4f}")
        return history
        
    def predict(self, x):    
        return self.forward(x, best_weights=True)

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


class ActivationFunction:
    def __init__(self, name):
        self.name = name

    def activate(self, x):
        if self.name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.name == 'relu':
            return np.maximum(0, x)
        elif self.name == 'linear':
            return x
        elif self.name == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError("Unsupported activation function")

    def derivative(self, x):
        if self.name == 'sigmoid':
            sig = self.activate(x)
            return sig * (1 - sig)
        elif self.name == 'relu':
            return (x > 0).astype(float)
        elif self.name == 'linear':
            return np.ones_like(x)
        elif self.name == 'softmax':
            return x * (1 - x) 
        else:
            raise ValueError("Unsupported activation function")