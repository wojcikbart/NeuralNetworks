import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from matplotlib.animation import FuncAnimation

class SelfOrganizingMap:
    def __init__(self, width, height, input_dim, init_method='random'):
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self._init_weights(init_method)
        self.history = []
        self._init_map()

    def _init_weights(self, method):
        if method == 'random':
            self.weights = np.random.rand(self.width, self.height, self.input_dim)
        elif method == 'grid':
            x = np.linspace(-1, 1, self.width)
            y = np.linspace(-1, 1, self.height)
            
            # Initialize the weights array
            self.weights = np.zeros((self.width, self.height, self.input_dim))
            
            # Set the first two dimensions as a grid
            for i in range(self.width):
                for j in range(self.height):
                    self.weights[i, j, 0] = x[i]  # X coordinate
                    self.weights[i, j, 1] = y[j]  # Y coordinate
                    
                    # If there are more dimensions, initialize them randomly
                    if self.input_dim > 2:
                        self.weights[i, j, 2:] = np.random.rand(self.input_dim - 2)



    def _init_map(self):
        self.grid_x, self.grid_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        self.grid_x = self.grid_x.flatten()
        self.grid_y = self.grid_y.flatten()
        self.grid = np.column_stack((self.grid_x, self.grid_y))
        self.grid = np.array(self.grid, dtype=np.float32)
        self.grid = MinMaxScaler().fit_transform(self.grid)

    def find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=2)
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_index
    
    def gaussian(self, dist, sigma):
        return np.exp(-dist**2 / (2 * sigma**2))

    def neg_second_gaussian(self, dist, sigma):
        return (dist**2 / sigma**4 - 1 / sigma**2) * np.exp(-dist**2 / (2 * sigma**2))

    
    def train(self, data, epochs, learning_rate, lambda_decay=10, sigma_param=1.0, dist_fun="gaussian"):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            alpha = learning_rate * np.exp(-epoch / lambda_decay)
            for x in data:
                bmu_index = self.find_bmu(x)
                bmu_coords = np.array(bmu_index)

                # Odległość siatki neuronów od BMU
                neuron_coords = np.dstack(np.meshgrid(np.arange(self.width), np.arange(self.height))).reshape(-1, 2)
                dists = np.linalg.norm(neuron_coords - bmu_coords, axis=1)

                # Funkcja sąsiedztwa
                if dist_fun == "gaussian":
                    neighborhood = self.gaussian(dists * sigma_param, sigma=1.0)
                elif dist_fun == "neg_gaussian":
                    neighborhood = self.neg_second_gaussian(dists * sigma_param, sigma=1.0)
                else:
                    neighborhood = np.ones_like(dists)

                # Aktualizacja wag
                for idx, (i, j) in enumerate(neuron_coords):
                    self.weights[i, j] += alpha * neighborhood[idx] * (x - self.weights[i, j])
            self.history.append(np.copy(self.weights))
            
    def plot_map(self, data):
        plt.figure(figsize=(8, 8))

        # Wagi neuronów - przekształcamy z (width, height, input_dim) do (n_neurons, input_dim)
        neuron_weights = self.weights.reshape(-1, self.input_dim)

        # Rysujemy punkty danych
        plt.scatter(data[:, 0], data[:, 1], c='blue', label='Dane')

        # Rysujemy wagi neuronów
        plt.scatter(neuron_weights[:, 0], neuron_weights[:, 1], c='red', marker='x', label='Neurony SOM')

        # Rysujemy linie między sąsiednimi neuronami (opcjonalnie)
        for i in range(self.width):
            for j in range(self.height):
                idx = i * self.height + j
                if i < self.width - 1:
                    right = (i+1) * self.height + j
                    plt.plot(
                        [neuron_weights[idx, 0], neuron_weights[right, 0]],
                        [neuron_weights[idx, 1], neuron_weights[right, 1]],
                        'gray', linewidth=0.5
                    )
                if j < self.height - 1:
                    below = i * self.height + (j+1)
                    plt.plot(
                        [neuron_weights[idx, 0], neuron_weights[below, 0]],
                        [neuron_weights[idx, 1], neuron_weights[below, 1]],
                        'gray', linewidth=0.5
                    )

        plt.title('Mapa wag neuronów SOM')
        plt.legend()
        plt.grid(True)
        plt.show()



    def plot_clusters(self, data):
        plt.figure(figsize=(10, 10))
        plt.scatter(data[:, 0], data[:, 1], c='blue', marker='o', label='Data Points')
        plt.scatter(self.grid[:, 0], self.grid[:, 1], c='red', marker='x', label='SOM Weights')
        plt.title('Clusters in Self-Organizing Map')
        plt.legend()
        plt.show()

    
def animate_som_training(som, data, interval=200, save_path=None):
    """
    Create an animation of the SOM training process.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot original data points
    ax.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.5, label='Training data')
    
    # Initialize SOM grid plot
    neuron_weights = som.history[0].reshape(-1, som.input_dim)
    scatter = ax.scatter(neuron_weights[:, 0], neuron_weights[:, 1], 
                          c='red', marker='x', s=80, label='SOM neurons')
    
    # Initialize grid lines
    lines = []
    for i in range(som.width):
        for j in range(som.height):
            idx = i * som.height + j
            
            # Horizontal connections
            if i < som.width - 1:
                right = (i+1) * som.height + j
                line, = ax.plot(
                    [neuron_weights[idx, 0], neuron_weights[right, 0]],
                    [neuron_weights[idx, 1], neuron_weights[right, 1]],
                    'gray', linewidth=0.5
                )
                lines.append(line)
                
            # Vertical connections
            if j < som.height - 1:
                below = i * som.height + (j+1)
                line, = ax.plot(
                    [neuron_weights[idx, 0], neuron_weights[below, 0]],
                    [neuron_weights[idx, 1], neuron_weights[below, 1]],
                    'gray', linewidth=0.5
                )
                lines.append(line)
    
    # Set plot properties
    ax.set_title('SOM Training Progress - Epoch 0')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Text for epoch counter
    epoch_text = ax.text(0.02, 0.98, 'Epoch: 0', transform=ax.transAxes, 
                          fontsize=12, verticalalignment='top')
    
    # Update function for animation
    def update(frame):
        neuron_weights = som.history[frame].reshape(-1, som.input_dim)
        
        # Update scatter plot
        scatter.set_offsets(neuron_weights[:, :2])
        
        # Update grid lines
        line_idx = 0
        for i in range(som.width):
            for j in range(som.height):
                idx = i * som.height + j
                
                # Horizontal connections
                if i < som.width - 1:
                    right = (i+1) * som.height + j
                    lines[line_idx].set_data(
                        [neuron_weights[idx, 0], neuron_weights[right, 0]],
                        [neuron_weights[idx, 1], neuron_weights[right, 1]]
                    )
                    line_idx += 1
                    
                # Vertical connections
                if j < som.height - 1:
                    below = i * som.height + (j+1)
                    lines[line_idx].set_data(
                        [neuron_weights[idx, 0], neuron_weights[below, 0]],
                        [neuron_weights[idx, 1], neuron_weights[below, 1]]
                    )
                    line_idx += 1
        
        # Update title and epoch counter
        ax.set_title(f'SOM Training Progress - Epoch {frame}')
        epoch_text.set_text(f'Epoch: {frame}')
        
        return [scatter] + lines + [epoch_text]
    
    # Create animation
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(som.history),
        interval=interval, 
        blit=True
    )
    
    plt.tight_layout()
    
    # Save animation if path is provided
    if save_path:
        anim.save(save_path, writer='pillow', fps=5)
    
    return anim


# Example usage
if __name__ == "__main__":
    data_hexagon = pd.read_csv('dane_koh/mio2/hexagon.csv')
    X_hexagon = data_hexagon[['x', 'y']]
    y_hexagon = data_hexagon['c']

    som = SelfOrganizingMap(width=4, height=4, input_dim=2, init_method='grid')
    som.train(X_hexagon.values, epochs=100, learning_rate=0.1, dist_fun="gaussian")
    anim = animate_som_training(som, X_hexagon.values, interval=1000)
    plt.show()