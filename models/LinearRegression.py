import numpy as np

class LinearRegression:
    def __call__(self, X):
        ''' Moves through a forward pass of the algorithm.
            Args: 
                X (np.array(shape=(n,m)): A matrix of n rows of examples with m feature columns
            Returns:
                np.array(shape=(n)): An real number output prediction for each example
                    '''
        X_transformed = np.hstack([X**i for i in range(1, self.degree+1)])
        return np.dot(X_transformed, self.weights) + self.bias
    
    def __init__(self, input_dim = 1, degree=1):
        self.weights = np.random.rand(input_dim*degree)
        self.bias = np.random.rand(1)
        self.degree = degree
        self.input_dim = input_dim
    
    def get_params(self):
        return np.concatenate((self.weights, self.bias))
    
    def set_params(self, params):
        split_idx = self.weights.size
        self.weights = params[:split_idx].reshape(self.weights.shape)
        self.bias = params[split_idx:].reshape(self.bias.shape)

    def grads(self, X):
        X_transformed = np.hstack([X**i for i in range(1, self.degree+1)])
        return np.hstack((X_transformed, np.reshape(np.ones(X.shape[0]), (X.shape[0], 1))))