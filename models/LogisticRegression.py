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
        return 1 / (1 + np.exp(-linear))
    
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
        y_pred = self(X)
        adjusted_loss_grads = loss_grad * (y_pred - y_pred**2) 
        return np.concatenate((np.dot(X.T, adjusted_loss_grads), [np.sum(adjusted_loss_grads)]))
    
class SoftMaxRegression:
    def __call__(self, X):
        ''' Moves through a forward pass of the algorithm.
            Args: 
                X (np.array(shape=(n,m)): A matrix of n rows of training examples with m feature columns
            Returns:
                np.array(shape=(m,k)): For each row example, a column represents the output probability w.r.t. that column's class.
                    i.e for the k'th column and i'th example, the value is the probability of the k'th class given the feature input of the i'th example
                    '''
        linears = np.matmul(X, self.weights) + self.biases
        return 1 / np.sum(np.exp(linears), axis=1) * linears
    def __init__(self, input_dim = 1, class_dim = 3, label_space = np.array([0, 1, 2])):
        self.weights = np.random.rand(input_dim, class_dim)
        self.biases = np.random.rand(class_dim)
        self.label_space = label_space
        self.label_mapping = {label: idx for idx, label in enumerate(label_space)}
        self.input_dim = input_dim
        self.class_dim = class_dim
    
    def get_params(self):
        return np.concatenate((np.reshape(self.weights, self.weights.size), self.biases))
    
    def set_params(self, params):
        split_idx = self.weights.size
        self.weights = params[:split_idx].reshape(self.weights.shape)
        self.bias = params[split_idx:].reshape(self.bias.shape)

    def grads(self, X, loss_grad):
        return np.matmul(loss_grad.T, X).T