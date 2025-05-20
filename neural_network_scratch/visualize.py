import matplotlib.pyplot as plt
import numpy as np
from neural_network_iris import NeuralNetwork

# visualize the training loss as the number of iterations increase
def plot_loss(costs):
    plt.figure(figsize=(8, 5))
    plt.plot(costs, label='Training Loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    costs = np.load("costs.npy")
    plot_loss(costs)