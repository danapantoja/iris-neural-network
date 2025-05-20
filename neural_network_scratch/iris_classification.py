from neural_network_iris import NeuralNetwork
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# load dataset
iris = load_iris()
X = iris.data
y = iris.target

# filter to only classes 0 and 1 (setosa and versicolor)
binary_mask = y < 2
X = X[binary_mask]
y = y[binary_mask]

# standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# reshape labels and split
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

nn = NeuralNetwork([4, 5, 1])  # 4 inputs → 5 hidden → 1 output
nn.train(X_train, y_train, epochs=10000, alpha=0.1)

# predict, shows accuracy at the end
preds = nn.predict(X_test)
accuracy = np.mean(preds == y_test)
print("Test Accuracy:", accuracy)
