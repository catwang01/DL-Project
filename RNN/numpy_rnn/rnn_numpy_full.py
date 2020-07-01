import numpy as np
from sklearn.preprocessing import OneHotEncoder

input_word = list("abcdefghijklmnopqrstuvwxyz")
input_word_onehot = OneHotEncoder(sparse=False).fit_transform(np.array(input_word).reshape(-1, 1))

# 序列长度取 3
sequenceLen = 3

x, y = [], []
for i in range(len(input_word_onehot) - sequenceLen):
    x.append(input_word_onehot[i:i + sequenceLen])
    y.append(input_word_onehot[i + 1: i + 1 + sequenceLen])

x_train = np.array(x)
y_train = np.array(y)

n_class = 26
nx = n_class
ny = n_class
nh = 4

def get_weights(shape, dtype=np.float32):
    np.random.seed(123)
    return np.array(np.random.randn(*shape), dtype=dtype)

def get_bias(shape, dtype=np.float32):
    return np.zeros(shape, dtype=dtype)

# 权重初始化，为了之后操作方便，放在一个字典中
weights = {
    'Wxh': get_weights((nx, nh)),
    'Why': get_weights((nh, ny)),
    'Whh': get_weights((nh, nh)),
    'bh': get_bias((1, nh)),
    'by': get_bias((1, ny))
}

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp
    return y

def xentropy(y, yhat):
    return np.mean(np.sum(-y * np.log(yhat + 1e-8), axis=0))

def _forward(xs, weights):
    Why = weights['Why']
    Whh = weights['Whh']
    Wxh = weights['Wxh']
    bh = weights['bh']
    by = weights['by']

    n_sequence = xs.shape[0]
    ny = Why.shape[1]
    nh = Wxh.shape[1]

    a = np.zeros((n_sequence, nh))
    h = np.zeros((n_sequence, nh))
    o = np.zeros((n_sequence, ny))
    yhat = np.zeros((n_sequence, ny))
    hprev = None

    for t, x in enumerate(xs):
        if t == 0:
            hprev = np.zeros((1, nh))
        else:
            hprev = h[t - 1]

        a[t] = np.matmul(x, Wxh) + np.matmul(hprev, Whh) + bh
        h[t] = np.tanh(a[t])
        o[t] = np.matmul(h[t], Why) + by
        yhat[t] = softmax(o[t])
    return yhat, a, h, o

def _backward(xs, ys, weights, a, o, h, yhat):
    n_sequences = xs.shape[0]

    Why = weights['Why']
    Whh = weights['Whh']
    Wxh = weights['Wxh']
    bh = weights['bh']
    by = weights['by']

    grads = { name: np.zeros_like(weights[name]) for name in weights}
    danext = None
    for i in range(n_sequences - 1, -1, -1):
        if i == n_sequences - 1:
            danext = np.zeros_like(a[i:i + 1])

        dot = yhat[i:i + 1] - ys[i:i + 1]

        # backprop through ot
        dby = dot
        dWhy = np.matmul(h[i:i + 1].T, dot)
        dht = np.matmul(dot, Why.T) + np.matmul(danext, Whh.T)
        dWhh = np.matmul(h[i:i + 1].T, danext)

        # backprop through ht
        dat = dht * (1 - h[i:i + 1] ** 2)

        # backprop through at
        dWxh = np.matmul(xs[i:i + 1].T, dat)
        dbh = dat

        # 累加梯度
        grads['by'] += dby
        grads['bh'] += dbh
        grads['Whh'] += dWhh
        grads['Wxh'] += dWxh
        grads['Why'] += dWhy
        danext = dat

    for k in grads:
        grads[k] = grads[k] / n_sequences
    return grads

def forward(batch_xs, weights):
    batch_yhat = []
    batch_a = []
    batch_o = []
    batch_h = []

    for xs in batch_xs:
        yhat, a, h, o = _forward(xs, weights)
        batch_yhat.append(yhat)
        batch_o.append(o)
        batch_h.append(h)
        batch_a.append(a)

    batch_yhat = np.array(batch_yhat)
    batch_a = np.array(batch_a)
    batch_h = np.array(batch_h)
    batch_o = np.array(batch_o)
    return batch_yhat, batch_a, batch_h, batch_o

def backward(batch_xs, batch_ys, weights, batch_a, batch_o, batch_h, batch_yhat):
    n_batch = batch_xs.shape[0]
    grads = { name: np.zeros_like(weights[name]) for name in weights}
    for xs, ys, a, o, h, yhat in zip(batch_xs, batch_ys, batch_a, batch_o, batch_h, batch_yhat):
        tmp_grads = _backward(xs, ys, weights, a, o, h, yhat)
        for k in tmp_grads:
            grads[k] += tmp_grads[k]
    for k in grads:
        grads[k] /= n_batch
    return grads


def sgd(grads, weights, lr=0.1):
    for name in weights:
        weights[name] -= lr * grads[name]

n_epochs = 100
for epoch in range(n_epochs):
    yhat, a, h, o = forward(x_train, weights)
    loss = xentropy(y_train, yhat)
    grads = backward(x_train, y_train, weights, a, o, h, yhat)
    sgd(grads, weights, lr=1)
    print(f"Epoch: {epoch}/{n_epochs} Train Loss: {loss}")



