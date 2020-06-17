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
x_dense_ = np.random.normal(size=(m, n_dense_features))
x_sparse_ = np.random.randint(0, 4, size=(m,))
encoder = OneHotEncoder(categories=[[0,1,2,3]], sparse=False)
# x_sparse_one_hot = encoder.fit_transform(x_sparse_)
# x_sparse_one_hot
# x_ = np.c_[x_sparse_, x_dense_]
y_ = np.random.normal(size=(m,))

x_sparse = tf.constant(x_sparse_, dtype=tf.int32)
x_dense = tf.constant(x_dense_, dtype=tf.float32)
y = tf.constant(y_, dtype=tf.float32)

w0 =  tf.Variable(0.0)
w1_dense = tf.Variable(np.random.normal(size=(n_dense_features, 1)), dtype=tf.float32)
w1_sparse = tf.Variable(np.random.normal(size=(n_class, 1)), dtype=tf.float32)
v_sparse = tf.Variable(np.random.normal(size=(n_class, k)), dtype=tf.float32)
v_dense = tf.Variable(np.random.normal(size=(n_dense_features, k)), dtype=tf.float32)

def forward(x_dense, x_sparse, w0, w1_dense, w1_sparse, v_dense, v_sparse):
    linear_dense = tf.squeeze(tf.matmul(x_dense, w1_dense))
    linear_sparse = tf.squeeze(tf.nn.embedding_lookup(w1_sparse, x_sparse))
    linear = w0 + linear_dense + linear_sparse

    mul_square = (tf.nn.embedding_lookup(v_sparse, x_sparse) + tf.matmul(x_dense, v_dense)) ** 2 
    square_mul = (tf.matmul(x_dense ** 2, v_dense**2) + tf.nn.embedding_lookup(v_sparse**2, x_sparse))

    y = linear + 1/2 * tf.reduce_mean(mul_square - square_mul , axis=1)
    return y

optizmier = tf.keras.optimizers.SGD(lr=0.001)

loss_list = []
n_epochs = 1000
for epoch in range(n_epochs):
    with tf.GradientTape() as tape:
        yhat = forward(x_dense, x_sparse, w0, w1_dense, w1_sparse, v_dense, v_sparse)
        mse = tf.reduce_mean(tf.losses.mse(y, yhat))
    grads = tape.gradient(mse, [w0, w1_dense, w1_sparse, v_dense, v_sparse])
    optizmier.apply_gradients(zip(grads, [w0, w1_dense, w1_sparse, v_dense, v_sparse]))
    loss_list.append(mse.numpy())

plt.plot(loss_list)
plt.title("Loss")
plt.show()

print(y)
print(forward(x_dense, x_sparse, w0, w1_dense, w1_sparse, v_dense, v_sparse))