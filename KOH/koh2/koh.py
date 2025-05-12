import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from sklearn.metrics import confusion_matrix
import seaborn as sns

class SelfOrganizingMap:
    def __init__(self, grid_shape, input_dim, topology='rectangular', neighborhood_fn='gaussian'):
        self.grid_shape = grid_shape  # (M, N)
        self.input_dim = input_dim
        self.topology = topology
        self.neighborhood_fn = neighborhood_fn
        self.weights = np.random.randn(*grid_shape, input_dim)
        
    def initialize_weights(self, X):
        n_samples = X.shape[0]
        size = np.prod(self.grid_shape)
        indices = np.random.choice(n_samples, size, replace=size > n_samples)
        self.weights = X[indices].reshape(self.grid_shape[0], self.grid_shape[1], -1)
    
    def find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)
    
    def get_grid_distance_matrix(self, bmu_i, bmu_j):
        i_indices, j_indices = np.indices(self.grid_shape)
        if self.topology == 'rectangular':
            distances = np.sqrt((i_indices - bmu_i)**2 + (j_indices - bmu_j)**2)
        elif self.topology == 'hexagonal':
            dq = j_indices - bmu_j
            dr = i_indices - bmu_i
            distances = (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) / 2
        return distances
    
    def train(self, X, num_epochs, beta, lambda_):
        self.initialize_weights(X)
        for epoch in range(num_epochs):
            lr = np.exp(-epoch / lambda_)
            np.random.shuffle(X)
            for x in X:
                bmu_i, bmu_j = self.find_bmu(x)
                distances = self.get_grid_distance_matrix(bmu_i, bmu_j)
                if self.neighborhood_fn == 'gaussian':
                    h = np.exp(-0.5 * (beta * distances)**2)
                elif self.neighborhood_fn == 'mexican_hat':
                    h = (1 - (beta * distances)**2) * np.exp(-0.5 * (beta * distances)**2)
                delta = lr * h[..., np.newaxis] * (x - self.weights)
                self.weights += delta
            print(f"Epoch {epoch+1}/{num_epochs} completed.")
    
    def map_vects(self, X):
        return np.array([self.find_bmu(x) for x in X])
    
    def plot_hexagonal_grid(self, values=None, title="SOM Grid"):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                x = j * 1.5
                y = i * np.sqrt(3) + (j % 2) * np.sqrt(3)/2
                
                hexagon = RegularPolygon((x, y), numVertices=6, radius=1.0,
                                       facecolor='white', edgecolor='black')
                ax.add_patch(hexagon)
                
                if values is not None:
                    plt.text(x, y, f"{values[i,j]:.2f}", ha='center', va='center')
        
        plt.title(title)
        plt.xlim(-1, self.grid_shape[1]*1.5)
        plt.ylim(-1, self.grid_shape[0]*np.sqrt(3))
        plt.show()
    
    def plot_rectangular_grid(self, values=None, title="SOM Grid"):
        plt.figure(figsize=(10, 10))
        
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                plt.plot(j, i, 'o', markersize=10, 
                        markeredgecolor='black', markerfacecolor='white')
                if values is not None:
                    plt.text(j, i, f"{values[i,j]:.2f}", ha='center', va='center')
        
        plt.title(title)
        plt.grid(True)
        plt.show()
    
    def analyze_clusters(self, X, y):
        bmus = self.map_vects(X)
        cluster_map = np.zeros((*self.grid_shape, len(np.unique(y))))
        
        for (i,j), label in zip(bmus, y):
            cluster_map[i,j,label] += 1
        
        # Plot cluster distribution
        plt.figure(figsize=(15, 5))
        for label in np.unique(y):
            plt.subplot(1, len(np.unique(y)), label+1)
            plt.imshow(cluster_map[:,:,label], cmap='viridis')
            plt.title(f"Class {label} distribution")
            plt.colorbar()
        plt.show()
        
        # Print purity statistics
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                total = np.sum(cluster_map[i,j])
                if total > 0:
                    dominant_class = np.argmax(cluster_map[i,j])
                    purity = cluster_map[i,j,dominant_class] / total
                    print(f"Neuron ({i},{j}): {total} samples, "
                          f"dominant class {dominant_class} (purity={purity:.2f})")
                    
if __name__ == "__main__":
    hexagon = pd.read_csv('dane_koh/mio2/hexagon.csv')
    X_hex = hexagon[['x', 'y']].values
    y_hex = hexagon['c'].values

    # Train SOM
    som = SelfOrganizingMap(grid_shape=(6, 6), input_dim=2, topology='hexagonal')
    som.train(X_hex, num_epochs=100, beta=1.0, lambda_=10)

    # Visualization
    som.plot_hexagonal_grid(title="Trained SOM (Hexagonal Topology)")
    som.plot_umatrix(title="U-Matrix for Hexagon Data")
    som.analyze_clusters(X_hex, y_hex)

    # Plot original data vs SOM mapping
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(X_hex[:,0], X_hex[:,1], c=y_hex, cmap='tab10')
    plt.title("Original Data")

    plt.subplot(122)
    bmus = som.map_vects(X_hex)
    for (i,j), label in zip(bmus, y_hex):
        x = j * 1.5
        y = i * np.sqrt(3) + (j % 2) * np.sqrt(3)/2
        plt.scatter(x, y, c=[label], cmap='tab10')
    plt.title("SOM Mapping")
    plt.show()