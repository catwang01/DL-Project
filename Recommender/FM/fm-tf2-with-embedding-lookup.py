import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np

m = 5   # sample size
n_sparse_features = 1
n_class = 4
n_dense_features = 6
n = n_sparse_features + n_dense_features # feature size
k = 3  # embedding size

np.random.seed(123)
x_dense = np.random.normal(size=(m, n_dense_features))

x_sparse = np.random.randint(0, 4, size=(m,1))
encoder = OneHotEncoder(categories=[[0,1,2,3]], sparse=False)
x_sparse_one_hot = encoder.fit_transform(x_sparse)
x_sparse_one_hot
x_ = np.c_[x_sparse, x_dense]
y_ = np.random.normal(size=(m,))

x = tf.constant(x_, dtype=tf.float32)
y = tf.constant(y_, dtype=tf.float32)
w0 =  tf.Variable(0.0)
w1 = tf.Variable(np.random.normal(size=(n, 1)), dtype=tf.float32)
v = tf.Variable(np.random.normal(size=(n, k)), dtype=tf.float32)

def forward(x, w0, w1, v):
    linear = w0 +  tf.squeeze(tf.matmul(x, w1))
    y =  linear + 1/2 * tf.reduce_mean(tf.matmul(x, v) ** 2 - tf.matmul(x ** 2, v ** 2), axis=1)
    return y

optizmier = tf.keras.optimizers.SGD(lr=0.001)

loss_list = []
n_epochs = 1000
for epoch in range(n_epochs):
    with tf.GradientTape() as tape:
        yhat = forward(x, w0, w1, v)
        mse = tf.reduce_mean(tf.losses.mse(y, yhat))
    grads = tape.gradient(mse, [w0, w1, v])
    optizmier.apply_gradients(zip(grads, [w0, w1, v]))
    loss_list.append(mse.numpy())

plt.ion()
plt.plot(loss_list)
plt.title("Loss")
plt.pause(0.05)
plt.close()

print(y)
print(forward(x, w0, w1, v))

# tf.Tensor([ 0.82711744  1.0707815  -3.0555775   1.005932   -1.7650565 ], shape=(5,), dtype=float32)
# tf.Tensor([ 0.8262203  1.0728326 -3.0549483  1.0054502 -1.7684705], shape=(5,), dtype=float32)

