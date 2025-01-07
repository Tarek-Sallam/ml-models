import numpy as np

class LinearRegression:
    def __init__(self, input_dim = 1, degree=1):
        self.weights = np.random.rand(input_dim*degree)
        self.bias = np.random.rand(1)
        self.degree = degree
        self.input_dim = input_dim

    def forward(self, X):
        X_transformed = np.hstack([X**i for i in range(1, self.degree+1)])
        return np.dot(X_transformed, self.weights) + self.bias
    
    def get_params(self):
        return np.concatenate((self.weights, self.bias))
    
    def set_params(self, params):
        split_idx = self.weights.size
        self.weights = params[:split_idx].reshape(self.weights.shape)
        self.bias = params[split_idx:].reshape(self.bias.shape)

    def grads(self, X, loss_grad):
        X_transformed = np.hstack([X**i for i in range(1, self.degree+1)])
        return np.concatenate((np.dot(X_transformed.T, loss_grad), [np.sum(loss_grad)]))
        