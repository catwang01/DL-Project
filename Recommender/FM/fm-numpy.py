import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf

n_samples = 5
n_features = 10
n_factors = 3

np.random.seed(123)
x = np.random.normal(size=(n_samples, n_features))
y = np.random.normal(size=(n_samples,))
w0 =  0.0
w1 = np.random.normal(size=(n_features, 1))
v = np.random.normal(size=(n_features, n_factors))

def forward(x, w0, w1, v):
    y = w0 + np.matmul(x, w1).squeeze() + 1/2 * np.sum(np.square(np.matmul(x, v)) - np.matmul(np.square(x), np.square(v)), axis=1)
    return y

def mse(y, yhat):
    return np.mean((y-yhat)**2)

def backward(yhat, y, batch_x, w0, w1, v):
    dyhat = -2 * (y - yhat)

    n_samples = batch_x.shape[0]
    n_factors = v.shape[1]

    dw0 = np.sum(dyhat) / n_samples
    dw1 = batch_x.T @ dyhat / n_samples
    dw1 = dw1[:, tf.newaxis]

    dv = np.zeros_like(v)
    for i in range(n_samples):
        x = batch_x[i][np.newaxis, :]
        dv += dyhat[i] * (x.T @ x @ v)
    dv -=  np.square(batch_x.T) @ dyhat[:, np.newaxis] @ np.ones(shape=[1, n_factors]) * v
    dv /= n_samples
    return dw0, dw1, dv

def apply_graidents(grads, variables, lr=0.001):
    for i in range(len(grads)):
        variables[i] -= lr * grads[i]

def gradient_check(x, y, w0, w1, v):
    if x.ndim == 1:
        x = x[np.newaxis, :]
    x = tf.constant(x, dtype=tf.float32)
    y = tf.constant(y, dtype=tf.float32)
    w0 = tf.Variable(w0, dtype=tf.float32)
    w1 = tf.Variable(w1, dtype=tf.float32)
    v = tf.Variable(v, dtype=tf.float32)
    with tf.GradientTape() as tape:
        yhat = w0 + tf.squeeze(tf.matmul(x, w1)) + 1 / 2 * tf.reduce_sum(tf.square(tf.matmul(x, v)) - tf.matmul(tf.square(x), tf.square(v)), axis=1)
        loss = tf.reduce_mean((y - yhat) ** 2)
    grads = tape.gradient(loss, [w0, w1, v])
    return grads

def check(x, y, w0, w1, v):
    yhat = forward(x, w0, w1, v)
    print(mse(y, yhat))
    grads = backward(yhat, y, x, w0, w1, v)
    grads_tf = gradient_check(x, y, w0, w1, v)
    print("Check whether gradients were computed correctly")
    print("Gradients computed: ", grads)
    print("Gradients compute by Tensorflow2: ", grads_tf)

# 检查梯度计算是否正确
check(x, y, w0, w1, v)

loss_list = []
n_epochs = 1000
for epoch in range(n_epochs):
    yhat = forward(x, w0, w1, v)
    loss_  = mse(y, yhat)
    grads = backward(yhat, y, x, w0, w1, v)
    apply_graidents(grads, [w0, w1, v])
    print(loss_)
    loss_list.append(loss_)

# 绘制loss曲线
plt.ion()
plt.plot(loss_list)
plt.title("Loss")
plt.pause(0.05)
plt.close()

# 输出结果查看准确性
print(y)
print(forward(x, w0, w1, v))


