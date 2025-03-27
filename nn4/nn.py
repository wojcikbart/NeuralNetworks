import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
class NeuralNetwork():
    def __init__(self, X, y, layers,
                  activation_fun='sigmoid', output_activation='linear',
                  loss_fun='mse', regularization=None, reg_lambda=0.001):
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
        self.activated_values = [0] * (len(self.layers) - 1)
        self.best_weights, self.best_biases = self.generate_random_weights(activation_fun=activation_fun)
        self.best_loss = np.inf
        self.model_age = 0
        self.best_age = 0
        self.loss_history = []

        # momentum
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.v_biases = [np.zeros_like(b) for b in self.biases]

        # rmsprop
        self.rmsprop_weights = [np.zeros_like(w) for w in self.weights]
        self.rmsprop_biases = [np.zeros_like(b) for b in self.biases]

    def generate_random_weights(self, activation_fun='sigmoid'):
        weights, biases = [], []
        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            fan_out = self.layers[i+1]
            if self.activation.name == 'sigmoid':
                limit = np.sqrt(6 / (fan_in + fan_out))  # Xavier for sigmoid
            elif self.activation.name == 'relu':
                limit = np.sqrt(2 / fan_in)  # He for ReLU
            weight_matrix = np.random.uniform(-limit, limit, (fan_in, fan_out))
            bias_vector = np.zeros((1, fan_out))  # Common to initialize biases to 0
            weights.append(weight_matrix)
            biases.append(bias_vector)
        return weights, biases

    def loss(self, y_true, y_pred, include_regularization=True):
        epsilon = 1e-15

        if self.loss_fun == 'mse':
            loss = np.mean(np.square(y_true - y_pred))
        elif self.loss_fun == 'mae':
            loss = np.mean(np.abs(y_true - y_pred))
        elif self.loss_fun == 'crossentropy':
            # Handle binary cross-entropy (single column output)
            if y_pred.shape[1] == 1:
                # Clip predictions to prevent log(0)
                y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
                loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            
            # Handle multi-class cross-entropy (multiple columns)
            else:
                # One-hot encoding check
                if y_true.ndim == 1 or y_true.shape[1] == 1:
                    # Convert to one-hot encoding if needed
                    y_true_onehot = np.zeros_like(y_pred)
                    for i, label in enumerate(y_true):
                        y_true_onehot[i, int(label)] = 1
                else:
                    y_true_onehot = y_true
                
                # Clip predictions to prevent log(0)
                y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
                loss = -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))

        # Add regularization if specified
        regularization = 0
        if self.regularization is not None and include_regularization:
            if self.regularization == 'l1':
                regularization = sum(np.sum(np.abs(w)) for w in self.weights)
            elif self.regularization == 'l2':
                regularization = sum(np.sum(np.square(w)) for w in self.weights)

        return loss + self.reg_lambda * regularization

    def loss_derivative(self, y_true, y_pred):
        if self.loss_fun == 'mse':
            return y_pred - y_true
        elif self.loss_fun == 'mae':
            return np.sign(y_pred - y_true)
        elif self.loss_fun == 'crossentropy':
            # F binary classification
            if y_pred.shape[1] == 1:
                return y_pred - y_true
            
            # Handle multi-class classification
            else:
                # One-hot encoding check
                if y_true.ndim == 1 or y_true.shape[1] == 1:
                    # Convert to one-hot encoding if needed
                    y_true_onehot = np.zeros_like(y_pred)
                    for i, label in enumerate(y_true):
                        y_true_onehot[i, int(label)] = 1
                else:
                    y_true_onehot = y_true
                
                return y_pred - y_true_onehot
    
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
            self.activated_values[i] = a
        return a
    
    def backward(self, x, y, y_pred, learning_rate, momentum, grad_threshold=10):
        error = self.loss_derivative(y, y_pred) * self.output_activation.derivative(y_pred)
        delta = error

        for i in reversed(range(len(self.layer_values))):
            if i < len(self.layer_values) - 1:  # For Hidden layers
                delta = np.dot(delta, self.weights[i + 1].T) * self.activation.derivative(self.layer_values[i])

            # Weights and biases update
            input = x if i == 0 else self.layer_values[i - 1]
            gradient = np.dot(input.T, delta) / x.shape[0]
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > grad_threshold:
                gradient = grad_threshold * gradient / grad_norm

            self.v_weights[i] = momentum * self.v_weights[i] + learning_rate * gradient
            self.weights[i] -= self.v_weights[i]
            
            
            self.v_biases[i] = momentum * self.v_biases[i] + learning_rate * np.mean(delta, axis=0, keepdims=True)
            self.biases[i] -= self.v_biases[i] 

    def backward_rmsprop(self, x, y, y_pred, learning_rate, decay, epsilon, grad_threshold=10):
        error = self.loss_derivative(y, y_pred) * self.output_activation.derivative(y_pred)
        delta = error

        for i in reversed(range(len(self.layer_values))):
            if i < len(self.layer_values) - 1:
                delta = np.dot(delta, self.weights[i + 1].T) * self.activation.derivative(self.layer_values[i])

            input = x if i == 0 else self.layer_values[i - 1]
            gradient = np.dot(input.T, delta) / x.shape[0]
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > grad_threshold:
                gradient = grad_threshold * gradient / grad_norm

            ## RMSprop
            self.rmsprop_weights[i] = decay * self.rmsprop_weights[i] + (1 - decay) * gradient ** 2
            adjusted_lr_weights = learning_rate / (np.sqrt(self.rmsprop_weights[i]) + epsilon)
            self.weights[i] -= adjusted_lr_weights * gradient

            self.rmsprop_biases[i] = decay * self.rmsprop_biases[i] + (1 - decay) * np.mean(delta, axis=0, keepdims=True) ** 2
            adjusted_lr_biases = learning_rate / (np.sqrt(self.rmsprop_biases[i]) + epsilon)
            self.biases[i] -= adjusted_lr_biases * np.mean(delta, axis=0, keepdims=True)

    def train(self, learning_rate, epochs, validation_data=None,
          mini_batch=False, batch_size=32, optimization='momentum',
          momentum=0, rmsprop_decay=0.9, epsilon=1e-8,
          stop_condition=0.5, report_interval=1000):
    
        x = self.X
        y = self.y
        num_samples = x.shape[0]
        wait = 0

        start_loss = self.loss(y, self.forward(x))
        print(f"Starting loss: {start_loss:.2f}")

        progress_bar = tqdm(
        range(epochs), 
        desc="Training", 
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
        ncols=150,
        smoothing=0.5,
        dynamic_ncols=True
        )


        try:
            for epoch in range(epochs):
                if mini_batch:
                    indices = np.random.permutation(num_samples)
                    for i in range(0, num_samples, batch_size):
                        batch_x = x[indices[i:i + batch_size]]
                        batch_y = y[indices[i:i + batch_size]]     
                        y_pred = self.forward(batch_x, store_values=True)
                        if optimization == 'momentum':
                            self.backward(batch_x, batch_y, y_pred, learning_rate, momentum)
                        elif optimization == 'rmsprop':
                            self.backward_rmsprop(batch_x, batch_y, y_pred, learning_rate, 
                                            rmsprop_decay, epsilon)
                else:
                    y_pred = self.forward(x, store_values=True)
                    if optimization == 'momentum':
                        self.backward(x, y, y_pred, learning_rate, momentum)
                    elif optimization == 'rmsprop':
                        self.backward_rmsprop(x, y, y_pred, learning_rate, 
                                            rmsprop_decay, epsilon)

                current_train_loss = self.loss(y, self.forward(x))

                if validation_data is not None:
                    x_val, y_val = validation_data
                    val_loss = self.loss(y_val, self.forward(x_val))
                    benchmark_loss = val_loss
                else:
                    benchmark_loss = current_train_loss

                if benchmark_loss < self.best_loss:
                    self.best_loss = benchmark_loss
                    self.best_weights = [w.copy() for w in self.weights]
                    self.best_biases = [b.copy() for b in self.biases]
                    self.best_age = self.model_age + epoch
                    wait = 0
                else:
                    wait += 1
                
                self.loss_history.append(benchmark_loss)
                
                progress_bar.set_postfix_str(f"Loss:  Current:{benchmark_loss:7.4f},Best:{self.best_loss:7.4f}", refresh=False)
                progress_bar.update()

                if benchmark_loss < stop_condition:
                    print(f"\nStopping condition met at epoch {self.model_age + epoch}")
                    break

        finally:
            progress_bar.close()

        self.model_age += epoch + 1
        print(f"\nTraining complete. Final loss: {self.best_loss:.4f}")
        
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
    
    def model_history(self):
        return self.loss_history

class ActivationFunction:
    def __init__(self, name):
        self.name = name

    def activate(self, x):
        if self.name == 'sigmoid':
            x = np.clip(x, -500, 500)
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
            sig = np.clip(self.activate(x), 1e-9, 1 - 1e-7)
            return sig * (1 - sig)
        elif self.name == 'relu':
            return (x > 0).astype(float)
        elif self.name == 'linear':
            return np.ones_like(x)
        elif self.name == 'softmax':
            s = self.activate(x)
            return s * (1 - s) 
        else:
            raise ValueError("Unsupported activation function")