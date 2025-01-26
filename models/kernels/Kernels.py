import numpy as np

class LinearKernel:
    def __call__(self, x, z):
        return np.dot(x, z)

class GaussianKernel:
    def __init__(self, sigma):
        self.sigma = sigma
    def __call__(self, x, z):
        return np.exp(-np.dot((x-z), (x-z))/(2*(self.sigma**2)))
    
class PolynomialKernel:
    def __init__(self, degree, constant):
        self.degree = degree
        self.constant = constant
    
    def __call__(self, x, z):
        return (np.dot(x, z) + self.constant) ** self.degree