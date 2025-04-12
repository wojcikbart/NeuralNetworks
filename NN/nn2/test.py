import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

class NeuralNetwork():
    def __init__(self, layers):
        """
        layers: list of integers, number of neurons in each layer (e.g. [2, 3, 1] for 2 input neurons, 3 neurons in the hidden layer, and 1 output neuron)
        This defines the necessary dimensions for the weight matrices and bias vectors.      
    """
        self.layers = layers
        self.weights = []
        self.biases = []
        self.weights, self.biases = self.generate_random_weights()
        self.layer_values = [0] * (len(self.layers) - 1)

    def generate_random_weights(self, weight_range=(0, 1)):
        w_min, w_max = weight_range
        weights = []
        biases = []
        for i in range(len(self.layers)- 1):
            weight_matrix = np.random.uniform(w_min, w_max, (self.layers[i], self.layers[i+1]))
            bias_vector = np.random.uniform(w_min, w_max, (1, self.layers[i+1]))
            weights.append(weight_matrix)
            biases.append(bias_vector)

        return weights, biases

    def activation(self, x):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def activation_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(a, w) + b
            a = self.activation(z) if i < len(self.weights) - 1 else z
            self.layer_values[i] = a  # Store activations for backprop
        return a
    
    def backward(self, x, y, y_pred, learning_rate):
        error = y - y_pred
        delta = error  # Gradient for output layer

        for i in reversed(range(len(self.layer_values))):
            if i < len(self.layer_values) - 1:  # Hidden layers
                delta = np.dot(delta, self.weights[i + 1].T) * self.activation_derivative(self.layer_values[i])

            # Weights and biases update
            input = x if i == 0 else self.layer_values[i - 1]
            self.weights[i] += learning_rate * np.dot(input.T, delta)
            self.biases[i] += learning_rate * np.mean(delta, axis=0, keepdims=True)


    
    # def backward(self, x, y, y_pred, learning_rate):
        
    #     error = y - y_pred    
    #     delta = error  # Start with output layer error

    #     for i in range(len(self.layer_values) - 1, -1, -1):  
    #         if i < len(self.layer_values) - 1:  # Hidden layers
    #             delta = np.dot(delta, self.weights[i + 1].T) * self.activation_derivative(self.layer_values[i])
    #         # print the shape of delta and layer_values[i]
    #         # print(delta.shape, self.layer_values[i-1].shape, self.weights[i].shape, self.biases[i].shape)

    #         # Compute weight update
    #         if i == 0:
    #             self.weights[i] += learning_rate * np.dot(x.T, delta)  
    #         else:
    #             self.weights[i] += learning_rate * np.dot(self.layer_values[i - 1].T, delta)  

    #         # Update biases
    #         self.biases[i] += learning_rate * np.mean(delta, axis=0, keepdims=True) 

    def train(self, x, y, learning_rate, epochs, mini_batch=False, batch_size=32, stop_condition=0.5):
        
        num_samples = x.shape[0]
        history = []
        pred_history = []

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

                if epoch % 100 == 0:
                    # print(f"Epoch {epoch}, MSE: {mse:.2f}")
                    pred_history.append(self.forward(x))

                if mse < stop_condition or not np.isfinite(mse):
                    break
        else:
            for epoch in range(epochs):
                y_pred = self.forward(x)
                mse = np.mean(np.square(y - y_pred))
                history.append(mse)
                if mse < stop_condition or not np.isfinite(mse):
                    break
                self.backward(x, y, y_pred, learning_rate)
                if epoch % 100 == 0:
                    # print(f"Epoch {epoch}, MSE: {mse:.2f}")
                    pred_history.append(y_pred)

        print(f"Final MSE: {mse}")

        self.animate_training(x, y, pred_history)
        return history
    
    def animate_training(self, x, y, pred_history):
    # Create the initial scatter plot
        fig, ax = plt.subplots()
        true_scatter = ax.scatter(x, y, color='blue', label='True values')
        pred_scatter = ax.scatter(x, pred_history[0], color='red', label='Predicted values')
        ax.legend()

        for pred in pred_history:
            # Update the predicted scatter plot
            pred_scatter.set_offsets(np.c_[x, pred])
            
            # Redraw the plot
            plt.draw()
            plt.pause(0.05)  # Pause for a short moment to animate
            # No need for plt.close() here because we are updating the same figure

        plt.show()
        
    def predict(self, x):
        y_pred = self.forward(x)
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

#  create main
if __name__ == "__main__":
    # data_step_train = pd.read_csv("dane_nn2\steps-small-training.csv")
    # data_step_train = data_step_train.drop(data_step_train.columns[0], axis=1)
    # data_step_train.head()
    # X_step_train = data_step_train['x'].values.reshape(-1, 1)
    # y_step_train = data_step_train['y'].values.reshape(-1, 1)

    # nn2 = NeuralNetwork([1, 10, 10, 1])
    # history_batch = nn2.train(X_step_train, y_step_train, 0.0002, int(1e5), mini_batch=True, batch_size=10, stop_condition=0.5)

    # # plt.plot(history, label="Full dataset")
    # plt.plot(history_batch, label="Mini-batch")
    # plt.xlabel("Epoch")
    # plt.ylabel("MSE")
    # plt.legend()
    # plt.show()

    data_square_train = pd.read_csv("dane_nn2\square-simple-training.csv")
    data_square_train = data_square_train.drop(data_square_train.columns[0], axis=1)
    data_square_train.head()
    X_square_train = data_square_train['x'].values.reshape(-1, 1)
    y_square_train = data_square_train['y'].values.reshape(-1, 1)

    # nn_square = NeuralNetwork([1, 10, 10, 1])
    # h1 = nn_square.train(X_square_train, y_square_train, 0.00004, int(1e5), stop_condition=0.7)

    nn_square2 = NeuralNetwork([1, 10, 10, 1])
    h2 = nn_square2.train(X_square_train, y_square_train, 0.00004, int(1e5), mini_batch=True, batch_size=32, stop_condition=0.7)

    # plt.plot(h1, label="Full dataset")
    plt.plot(h2, label="Mini-batch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()