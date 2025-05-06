import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_som_training(som, data, interval=200, save_path=None):
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
    ax.set_title('SOM Training Progress')
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
        ax.set_title(f'SOM Training Progress')
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
