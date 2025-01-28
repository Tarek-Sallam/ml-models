import numpy as np

class SupportVectorMachine:
    def __call__(self, X):
        if (self.mode=="dual"):
            if (self.alphas):
                np.sum(self.alphas * self.support_labels * self.kernel(self.support_vectors, X)) + self.bias
            else:
                print("Model not trained yet.")
        else:
            return [1 if np.dot(self.weights, x) + self.bias >= 0 else -1 for x in X]
            

    def __init__(self, input_dim, mode="primal", kernel=None, C=0):
        self.kernel = kernel
        self.input_dim = input_dim
        self.mode = mode
        self.C = C
        self.bias = np.random.rand(1)
        if self.mode == "primal":
            self.weights = np.random.rand(input_dim)
        else:
            self.alphas = None
            self.support_vectors = None
            self.support_labels = None
            if self.C == 0:
                self.hard = True
        
    def get_params(self):
        if self.mode == "primal":
            return np.concatenate((self.weights, self.bias))
        else:
            return self.alphas
        
    def set_params(self, params, y=None):
        if self.mode == "primal":
            split_idx = self.weights.size
            self.weights = params[:split_idx].reshape(self.weights.shape)
            self.bias = params[split_idx:].reshape(self.bias.shape)
        else:
            self.alphas = params
            while True:
                self.alphas = np.clip(self.alphas, 0, self.C)
                rand_idx = np.random.randint(0, self.alphas.shape(0))
                all_alphas = (self.alphas * y)
                all_alphas = all_alphas[:rand_idx] + all_alphas[rand_idx+1:]
                new_alpha = -1/y[rand_idx]*np.sum(all_alphas)
                self.alphas[rand_idx] = new_alpha
                if new_alpha >= 0 and new_alpha <= self.C:
                    break

    def set_supports(self, X, y):
        if self.mode == "primal":
            return
        else:
            self.support_vectors = []
            self.support_labels = []
            alphas = []
            for i in range(self.alphas):
                if (self.alphas[i] > 0 and self.hard) or (self.alphas[i] > 0 and self.alphas[i] < self.C and not self.hard):
                    self.support_vectors.append(X[i])
                    self.support_labels.append(y[i])
                    alphas.append(self.alphas[i])
            
            self.support_labels = np.array(self.support_labels)
            self.support_vectors = np.array(self.support_vectors)
            self.alphas = np.array(alphas)
            self.bias = np.mean(self.alphas * self.support_labels * self.kernel(self.support_vectors, X))

    def grads(self, X, y):
        if self.mode == "primal":
            return np.hstack((X, np.reshape(np.ones(X.shape[0]), (X.shape[0], 1))))
        else:
            if not self.alphas:
                self.alphas = np.random.rand(y.shape)
            return np.matmul(self.kernel(X.T, X.T).T, (y*self.alphas)) * y - np.ones_like(y)