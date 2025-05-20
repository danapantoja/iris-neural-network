
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.L = len(layer_sizes) - 1  # number of layers excluding input
        self.n = layer_sizes
        self.W = {}
        self.b = {}

        #initialize weights and biases, layer 0 has no weights or bias matrix
        #weight matrices (n[l] x n[l-1])
        #bias matrices (vectors) (n[1] x 1)
        for l in range(1, self.L + 1):
            self.W[l] = np.random.randn(self.n[l], self.n[l-1])
            self.b[l] = np.random.randn(self.n[l], 1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def feed_forward(self, A0):
        # dictionary: A stores activation nodes at each layer, A[0] = A0
        A = {0: A0}

        # dictionary : stores the (weight * node value + bias) at each node (before the activation)
        Z = {}

        # for each layer, we do the calculations for Z ^l(matrix containing the values of each node)
        # for each layer, after the Z calculation, we change these values in the A^l matrix, using activation function
        for l in range(1, self.L + 1):
            Z[l] = self.W[l] @ A[l-1] + self.b[l]
            A[l] = self.sigmoid(Z[l])

        # returns the activation matrix for the final layer, the A dictionary (every layer), and Z dictionary (every layer)
        return A[self.L], A, Z
    
    #using sigmoid function for the activation (keeps the values in between 0 and 1)
    def sigmoid_derivative(self, A):
        return A * (1 - A)

    #from output (A^L) to (A^0), we calculate partial derivatives to get the Cost gradient,
    # which we can then use the negative gradient to get to a minimum cost

    # takes in y_hat (A^L), Y (correct output), Z (node matrices for each layer), and A
    def backprop(self, y_hat, Y, A, Z, m):

        #dictionary stores all the partial derivaties for weight, bias, A, and Z for each layer
        
        #dC/dW^l
        dW = {}
        #dC/dB^l
        db = {}
        #dC/dA^l
        dA = {}
        #dC/dZ^l
        dZ = {}

        # Output layer partial derivative matrices
        
        #dC/dZ^L = dC/dA^L @ dA^L/ dZ^L = 1/m * A^L - Y, and A^L = y_hat
        dZ[self.L] = (1/m) * (y_hat - Y)
        # dC/dW^L = dC/dZ^L @ A^L-1 ^T (transpose to get right dimensions)
        dW[self.L] = dZ[self.L] @ A[self.L - 1].T
        #making it n[l] x 1 by compressing to make it one column
        db[self.L] = np.sum(dZ[self.L], axis=1, keepdims=True)

        # hidden layers gradients
        for l in range(self.L - 1, 0, -1):
            #using the derivative from previous layer to get the dA for this layer
            # dC/ dA^l = dC/dZ^l+1 * dZ^l+1/d^A^l, and dZ^l+1/d^A^l = W^l+1 !!!!!,
            #  transpose and make W^l+1 the first term to get the right dimensions
            dA[l] = self.W[l+1].T @ dZ[l+1]
            # dC/ dZ^l = dC/ dA^l * dA^l / dZ^l , we already calculated dC/dA^L, now we need dA^l / dZ^l, which 
            # is calculated by getting the derivative of the sigmoid function
            dZ[l] = dA[l] * self.sigmoid_derivative(A[l])

            # dC/dW^l = dC/ dZ * dZ/ dW^l, and dZ/ dW^l = A^l-1, transpose to get the right dimensions
            dW[l] = dZ[l] @ A[l-1].T
            db[l] = np.sum(dZ[l], axis=1, keepdims=True)
        #outputs the dC/dW^1, dC/db^1
        return dW, db
    
    # using the binary cross entopy loss function, takes truth values (Y) and predicted output (y_hat)
    def cost(self, y_hat, Y):
        # if Y = 1, then we have log(y_hat), if y_hat is 1, the loss is 0, (good)
        # if Y = 1, and log(y_hat) = 0, the loss is 1 (bad)
        # similar when Y=0
        losses = - (Y * np.log(y_hat) + (1 - Y) * np.log(1 - y_hat))
        #returns the mean of all the losses
        return np.sum(losses) / Y.shape[1]


    #takes in input data, truth labels, number of iterations for the feedforward process, 
    # and the size of the step to reach the minimum cost for each iteration
    def train(self, X, Y, epochs=10000, alpha=0.1):

        #A0 is layer 0, we transpose so every column is an input sample 
        A0 = X.T
        # number of training samples
        m = X.shape[0]
        # make the Y (truth labels) match the output layer dimensions
        Y = Y.reshape(self.n[-1], m)
        #stores the cost value at each iteration
        costs = []

        # for each feed forward
        for e in range(epochs):
            # change the values according to new weights and bias values
            y_hat, A, Z = self.feed_forward(A0)
            #gets the cost
            cost = self.cost(y_hat, Y)
            #adds to cost array
            costs.append(cost)
            #changes the dW^1, and db^1 
            dW, db = self.backprop(y_hat, Y, A, Z, m)

            # then for each layer, changes the weights and biases to get a little bit closer to the
            # cost function minimum for the entire thing
            for l in range(1, self.L + 1):
                self.W[l] -= alpha * dW[l]
                self.b[l] -= alpha * db[l]

            #outputs the cost every 100 iterations
            if e % 100 == 0:
                print(f"Epoch {e}: Cost = {cost:.4f}")
        
        np.save("costs.npy", costs)
        return costs
    
    def predict(self, X):
        A0 = X.T
        y_hat, _, _ = self.feed_forward(A0)
        return (y_hat > 0.5).astype(int).T

