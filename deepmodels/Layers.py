import numpy as np

class Activation():
    def has_params():
        return False
    
class ParamLayer():
    def has_params():
        return True
    
class Linear(ParamLayer):
    def __init__(self, input_dim, layer_dim):
        self.input_dim = input_dim
        self.layer_dim = input_dim
        self.layer_dim = input_dim
        self.weights = np.ones((layer_dim, input_dim))
        self.bias = np.zeros(layer_dim)

    def __call__(self, X):
        self.forward(X)

    def get_params(self):
        return self.weights + np.vstack((self.weights, self.bias))
    
    def forward(self, X):
        return (self.weights @ X) + self.bias
    
class MLP:
    def __init__(self, layers):
        self.layers = []
        for layer in layers:
            self.add(layer)

    def add(self, layer):
        self.layers.append(layer)

    def get_params(self):
        all_params = np.array([], dtype=object)
        for layer in self.layers:
            if layer.has_params():
                if (all_params.size == 0):
                    all_params = params
                params = layer.get_params()
                all_params = all_params.append(params)
        
        return all_params

    def __call__(self, X):
        self.forward(X)

    def forward(self, X):
        layer_output = np.copy(X)
        for layer in self.layers:
            layer_output = layer(layer_output)

        return layer_output


class ReLU(Activation):
    def __call__(self, x):
        return np.clip(x, a_min=0)
    
class Tanh(Activation):
    def __call__(self, x):
        return np.tanh(x)

class Sigmoid(Activation):
    def __call__(self, x):
        return 1 / (1 + np.e**-x)
    
