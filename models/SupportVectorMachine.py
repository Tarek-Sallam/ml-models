import numpy as np

class SupportVectorMachine:
    def __call__(self, X, y):
        if (self.alphas):
            bias = 1; # edit later
            np.sum(self.alphas * self.support_labels * self.kernel(self.support_vectors, X)) + self.bias

    def __init__(self, input_dim, kernel):
        self.kernel = kernel
        self.input_dim = input_dim
        self.alphas = None
        self.support_vectors = None
        self.support_labels = None
    

