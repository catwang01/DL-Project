import numpy as np
from matplotlib import pyplot as plt

R = np.array([[5, 5, 0, 5], [5, 0, 3, 4], [3, 4, 0, 3], [0, 0, 5, 3],
              [5, 4, 4, 5], [5, 4, 5, 5]])

n = 6
m = 4
k = 3
np.random.seed(123)
U = np.random.randn(n, k)
V = np.random.randn(m, k)

def loss(R, U, V, lambda_=0.01):
    mask = R > 0
    l = ((R - np.matmul(U, V.T))**2 * mask).sum()
    l += lambda_ * (np.linalg.norm(U, ord=2, axis=0)**2).sum()
    l += lambda_ * (np.linalg.norm(V, ord=2, axis=0)**2).sum()
    return l


def update(R, U, V, lr=0.02, lambda_=0.01):
    nrow, ncol = R.shape[0], R.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            if R[i][j] != 0:
                error = R[i][j] - np.dot(U[i], V[j])
                dui = -2 * error * V[j] + 2 * lambda_ * U[i]
                dvj = -2 * error * U[i] + 2 * lambda_ * V[j]
                U[i] -= lr * dui
                V[j] -= lr * dvj


loss_list = []
n_epochs = 100
for epoch in range(n_epochs):
    update(R, U, V)
    loss_list.append(loss(R, U, V))

plt.plot(loss_list)
plt.title("loss")
plt.show()

print(U)
print(V)
print(R)
print(np.matmul(U, V.T))