import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from som_utils import animate_som_training
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
            self.weights = np.zeros((self.width, self.height, self.input_dim))
            for i in range(self.width):
                for j in range(self.height):
                    self.weights[i, j, 0] = x[i]
                    self.weights[i, j, 1] = y[j]
                    if self.input_dim > 2:
                        self.weights[i, j, 2:] = np.random.rand(self.input_dim - 2)

    def _init_map(self):
        self.grid_x, self.grid_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        self.grid = np.column_stack((self.grid_x.flatten(), self.grid_y.flatten()))
        self.grid = MinMaxScaler().fit_transform(self.grid)

    def find_bmu(self, x):
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def gaussian(self, dist, sigma):
        return np.exp(-dist**2 / (2 * sigma**2))

    def neg_second_gaussian(self, dist, sigma):
        return (dist**2 / sigma**4 - 1 / sigma**2) * np.exp(-dist**2 / (2 * sigma**2))

    def train(self, data, epochs, learning_rate, lambda_decay=10, sigma_param=1.0, dist_fun="gaussian"):

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            alpha = learning_rate * np.exp(-epoch / lambda_decay)
            sigma = sigma_param * np.exp(-epoch / lambda_decay)

            for x in data:
                bmu_index = self.find_bmu(x)
                bmu_coords = np.array(bmu_index)

                neuron_coords = np.dstack(np.meshgrid(np.arange(self.width), np.arange(self.height))).reshape(-1, 2)
                dists = np.linalg.norm(neuron_coords - bmu_coords, axis=1)

                if dist_fun == "gaussian":
                    neighborhood = self.gaussian(dists, sigma)
                elif dist_fun == "neg_gaussian":
                    neighborhood = self.neg_second_gaussian(dists, sigma)
                else:
                    neighborhood = np.ones_like(dists)

                for idx, (i, j) in enumerate(neuron_coords):
                    self.weights[i, j] += alpha * neighborhood[idx] * (x - self.weights[i, j])

            self.history.append(np.copy(self.weights))

    def neuron_class_counts(self, data, labels):
        from collections import defaultdict
        counts = defaultdict(lambda: defaultdict(int))
        for x, label in zip(data, labels):
            bmu = self.find_bmu(x)
            counts[bmu][label] += 1
        return counts
            
    def plot_map(self, data):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        neuron_weights = self.weights.reshape(-1, self.input_dim)

        if self.input_dim == 2:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(data[:, 0], data[:, 1], c='blue', s=10, label='Dane')
            ax.scatter(neuron_weights[:, 0], neuron_weights[:, 1], c='red', marker='x', s=80, linewidths=2, label='Neurony SOM')

            for i in range(self.width):
                for j in range(self.height):
                    idx = i * self.height + j
                    if i < self.width - 1:
                        right = (i + 1) * self.height + j
                        ax.plot([neuron_weights[idx, 0], neuron_weights[right, 0]],
                                [neuron_weights[idx, 1], neuron_weights[right, 1]],
                                color='black', linewidth=1.5)
                    if j < self.height - 1:
                        below = i * self.height + (j + 1)
                        ax.plot([neuron_weights[idx, 0], neuron_weights[below, 0]],
                                [neuron_weights[idx, 1], neuron_weights[below, 1]],
                                color='black', linewidth=1.5)

            ax.set_title('2D SOM Map')
            ax.legend()
            ax.grid(True)

        elif self.input_dim == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', s=10, label='Dane')
            ax.scatter(neuron_weights[:, 0], neuron_weights[:, 1], neuron_weights[:, 2],
                    c='red', marker='x', s=80, linewidths=2, label='SOM neurons')

            for i in range(self.width):
                for j in range(self.height):
                    idx = i * self.height + j
                    if i < self.width - 1:
                        right = (i + 1) * self.height + j
                        ax.plot([neuron_weights[idx, 0], neuron_weights[right, 0]],
                                [neuron_weights[idx, 1], neuron_weights[right, 1]],
                                [neuron_weights[idx, 2], neuron_weights[right, 2]],
                                color='black', linewidth=1.5)
                    if j < self.height - 1:
                        below = i * self.height + (j + 1)
                        ax.plot([neuron_weights[idx, 0], neuron_weights[below, 0]],
                                [neuron_weights[idx, 1], neuron_weights[below, 1]],
                                [neuron_weights[idx, 2], neuron_weights[below, 2]],
                                color='black', linewidth=1.5)

            ax.set_title('3D SOM Map')
            ax.legend()

        else:
            raise ValueError("Only 2D and 3D data are supported for plotting.")

        plt.show()


    def assign_labels(self, data):
        if not hasattr(self, 'cluster_labels'):
            raise ValueError("First call assign_clusters(data, labels), aby przypisać klasy do neuronów.")

        assigned_labels = []
        for x in data:
            bmu = self.find_bmu(x)
            label = self.cluster_labels.get(bmu, -1)  # Jeśli neuron nie ma przypisanej etykiety, daj -1
            assigned_labels.append(label)

        return np.array(assigned_labels)
    
    def assign_clusters(self, data, labels):
        from collections import defaultdict, Counter

        # Zliczamy przypisania etykiet do neuronów
        label_map = defaultdict(list)
        for x, label in zip(data, labels):
            bmu = self.find_bmu(x)
            label_map[bmu].append(label)

        # Dla każdego neuronu wybieramy klasę większościową
        self.cluster_labels = {}
        for bmu_coords, labels in label_map.items():
            most_common = Counter(labels).most_common(1)[0][0]
            self.cluster_labels[bmu_coords] = most_common

        return self.cluster_labels.values()
    
    def plot_map_colored_by_cluster(self):
        if not hasattr(self, 'cluster_labels'):
            raise ValueError("First call assign_clusters(data, labels).")

        plt.figure(figsize=(8, 8))

        all_labels = list(set(self.cluster_labels.values()))
        label_to_color = {label: idx for idx, label in enumerate(all_labels)}
        cmap = plt.get_cmap('tab10', len(all_labels))

        for i in range(self.width):
            for j in range(self.height):
                coord = (i, j)
                weight = self.weights[i, j]
                if coord in self.cluster_labels:
                    label = self.cluster_labels[coord]
                    color = cmap(label_to_color[label])
                else:
                    color = 'lightgray'

                plt.scatter(weight[0], weight[1], color=color, s=100, edgecolors='k')

        plt.title("SOM with Cluster Labels")
        plt.grid(True)
        plt.show()

    def plot_clusters(self, data):
        plt.figure(figsize=(10, 10))
        plt.scatter(data[:, 0], data[:, 1], c='blue', marker='o', label='Data Points')
        plt.scatter(self.grid[:, 0], self.grid[:, 1], c='red', marker='x', label='SOM Weights')
        plt.title('Clusters in Self-Organizing Map')
        plt.legend()
        plt.show()

    def compare_with_real_labels(self, data, real_labels):
        correct = 0
        total = len(data)

        for x, real_label in zip(data, real_labels):
            bmu = self.find_bmu(x)
            
            predicted_label = self.cluster_labels.get(bmu, None)
            
            if predicted_label == real_label:
                correct += 1

        accuracy = correct / total
        return accuracy, correct, total    


if __name__ == "__main__":
    data_hexagon = pd.read_csv('dane_koh/mio2/hexagon.csv')
    X_hexagon = data_hexagon[['x', 'y']]
    y_hexagon = data_hexagon['c']

    som = SelfOrganizingMap(width=5, height=5, input_dim=2, init_method='grid')
    som.train(X_hexagon.values, epochs=25, learning_rate=0.01, dist_fun="gaussian")
    anim = animate_som_training(som, X_hexagon.values, interval=500)
    plt.show()
    som.assign_clusters(X_hexagon.values, y_hexagon.values)
    som.plot_map_colored_by_cluster()
    accuracy, correct, total = som.compare_with_real_labels(X_hexagon.values, y_hexagon.values)

    print(f"Dokładność klasyfikacji: {accuracy * 100:.2f}%")
    print(f"Liczba poprawnych klasyfikacji: {correct}/{total}")