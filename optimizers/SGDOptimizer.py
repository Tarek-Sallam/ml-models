class SGDOptimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, params, gradients):
        params -= self.learning_rate * gradients