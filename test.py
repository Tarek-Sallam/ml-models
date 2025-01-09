from Trainer import Trainer
from optimizers.SGDOptimizer import SGDOptimizer
from models.LogisticRegression import LogisticRegression
from models.LinearRegression import LinearRegression
from loss.CrossEntropyLoss import BinaryCrossEntropyLoss
from loss.MSELoss import MSELoss
import numpy as np
import matplotlib.pyplot as plt

# GENERATE DATA
# x = np.random.rand(40)
# y = np.round(x + (np.random.rand() - 0.5) * 0.1 + 0.1, 0)
# X = np.reshape(x, (x.shape[0],1))

x = np.linspace(0, 10, 10)
y = x + 2 + np.random.rand() - 0.5
#x2 = np.linspace(0, 1, 40)
#X2 = np.reshape(x2, (x2.shape[0],1))
X = np.reshape(x, (x.shape[0], 1))
model = LinearRegression(1, 1)
optimizer = SGDOptimizer(0.01)
loss = MSELoss()
trainer = Trainer(model, loss, optimizer)

trainer.train(X, y, 10)
fig, ax = plt.subplots()

ax.plot(x, y, 'o')
y_pred = trainer.model(X)
ax.plot(x, y_pred)
plt.show()