import numpy as np

class LogisticRegression:
    def __call__(self, X):
        ''' Moves through a forward pass of the algorithm.
            Args: 
                X (np.array(shape=(n,m)): A matrix of n rows of training examples with m feature columns
            Returns:
                np.array(shape=(n)): An output probability of classifying a 1 for each training example
                    '''
        linear = np.dot(X, self.weights) + self.bias
        return 1 / (1 + np.e**(-linear))
    
    def __init__(self, input_dim = 1):
        self.weights = np.random.rand(input_dim)
        self.bias = np.random.rand(1)
        self.input_dim = input_dim

    def get_params(self):
        return np.concatenate((self.weights, self.bias))
    
    def set_params(self, params):
        split_idx = self.weights.size
        self.weights = params[:split_idx].reshape(self.weights.shape)
        self.bias = params[split_idx:].reshape(self.bias.shape)

    def grads(self, X, loss_grad):
        return np.concatenate((np.dot(X.T, loss_grad), [np.sum(loss_grad)]))
    
class SoftMaxRegression:
    def __call__(self, X):
        pass
    def __init__(self, input_dim = 1, class_dim = 3):
        self.weights = np.random.rand(input_dim, class_dim)
        self.biases = np.random.rand(class_dim)
        self.input_dim = input_dim
        self.class_dim = class_dim

    def __call__(self, X):
        ''' Moves through a forward pass of the algorithm.
            Args: 
                X (np.array(shape=(n,m)): A matrix of n rows of training examples with m feature columns
            Returns:
                np.array(shape=(n)): An output probability of classifying a 1 for each training example
                    '''
        linears = np.dot()
        return 1 / (1 + np.e**(-linear))
