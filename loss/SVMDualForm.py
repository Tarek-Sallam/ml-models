import numpy as np

class SVMDualForm:
    def __call__(self, X, alphas):
        pass
    def __init__(self, C):
        self.C = C

    def grads(self, X, y, kernel):
        kernel(X.T, X.T).T
        
