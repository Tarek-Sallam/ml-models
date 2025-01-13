import numpy as np

class SupportVectorMachine:
    def __call__(self, X, y):
        pass
        
    def __init__(self, kernel):
        self.kernel = kernel
        self.alphas = None

    def get_params(self):
        return self.alphas
    
    def set_params(self, params):
        self.alphas = params

    def grads(self):
        pass