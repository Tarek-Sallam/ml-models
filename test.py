from Trainer import Trainer
from optimizers.SGDOptimizer import SGDOptimizer
from models.Regression import LinearRegression
from loss.MSELoss import MSELoss
import numpy as np
import matplotlib.pyplot as plt

# GENERATE DATA
x = np.linspace(-1, 1, 9)
y = np.random.rand(9) * 100

x2 = np.linspace(-1, 1, 200)
X2 = x2.reshape((x2.shape[0],1))
X = x.reshape((x.shape[0],1))

model = LinearRegression(1, 20)
optimizer = SGDOptimizer(0.01)
loss = MSELoss()
trainer = Trainer(model, loss, optimizer)

trainer.train(X, y, 10000)
fig, ax = plt.subplots()

ax.plot(x, y, 'o')
y_pred = trainer.model.forward(X2)
ax.plot(x2, y_pred)
plt.show()