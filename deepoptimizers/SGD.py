class SGDOptimizer:
    def __init__(self, params, learning_rate=0.01, ):
        self.learning_rate = learning_rate
        self.trainable_params = params

    def step(self, gradients):
        params -= self.learning_rate * gradients