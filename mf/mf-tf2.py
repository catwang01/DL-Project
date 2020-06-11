import tensorflow as tf 
import numpy as np
from matplotlib import pyplot as plt

R = tf.constant([[5, 5, 0, 5], [5, 0, 3, 4], [3, 4, 0, 3], [0, 0, 5, 3],
              [5, 4, 4, 5], [5, 4, 5, 5]], dtype=tf.float32)

m = 6
n = 4
k = 3
lr = 0.02
np.random.seed(123)
U = tf.Variable(np.random.randn(m, k), dtype=tf.float32)
V = tf.Variable(np.random.randn(n, k), dtype=tf.float32)
mask = R > 0


# 定义损失函数
def mse(R, U, V, mask, lambda_=0.01):
    l = tf.reduce_sum(((R - tf.matmul(U, tf.transpose(V)))**2 * tf.cast(mask, tf.float32)))
    l += tf.reduce_sum(lambda_ * (tf.linalg.norm(U, ord=2, axis=0)**2))
    l += tf.reduce_sum(lambda_ * (tf.linalg.norm(V, ord=2, axis=0)**2))
    return l

# 定义优化器
optimizer = tf.keras.optimizers.SGD(
    learning_rate=lr, momentum=0.0, nesterov=False
)

# 训练11步
n_epochs = 100
loss_list = []
for epoch in range(n_epochs):
    with tf.GradientTape() as tape:
        loss = mse(R, U, V, mask)
    grads = tape.gradient(loss, [U, V])
    optimizer.apply_gradients(zip(grads, [U, V]))
    loss_list.append(loss.numpy())


# 打印loss的变化
plt.plot(loss_list)
plt.title('loss')
plt.show()

print(U)
print(V)
print(R)
print(tf.matmul(U, tf.transpose(V)))
