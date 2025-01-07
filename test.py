from Trainer import Trainer
from optimizers.SGDOptimizer import SGDOptimizer
from models.LogisticRegression import LogisticRegression
from loss.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss
import numpy as np
import matplotlib.pyplot as plt

# GENERATE DATA
x = np.random.rand(40)
y = np.round(x + (np.random.rand() - 0.5) * 0.1 + 0.1, 0)
X = np.reshape(x, (x.shape[0],1))

x2 = np.linspace(0, 1, 40)
X2 = np.reshape(x2, (x2.shape[0],1))

model = LogisticRegression(1)
optimizer = SGDOptimizer(0.05)
loss = BinaryCrossEntropyLoss()
trainer = Trainer(model, loss, optimizer)

print(trainer.model.get_params())
print(model.forward(X2))
trainer.train(X, y, 10000)
print(model.forward(X2))
fig, ax = plt.subplots()

ax.plot(x, y, 'o')
y_pred = trainer.model.forward(X2)
ax.plot(x2, y_pred)
plt.show()