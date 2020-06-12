import tensorflow as tf
from matplotlib import pyplot as plt

m = 5
n = 10
k = 3

tf.random.set_seed(123)
x = tf.random.normal(shape=(m, n))
y = tf.random.normal(shape=(m, ))
w0 =  tf.Variable(0.0)
w1 = tf.Variable(tf.random.normal(shape=(n, 1)))
v = tf.Variable(tf.random.normal(shape=(n, k)))

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

