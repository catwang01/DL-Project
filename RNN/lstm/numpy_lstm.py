#!/usr/bin/env python
# coding: utf-8

# [toc]
# 
# # Numpy LSTM

# In[62]:


import tensorflow as tf

tf.random.set_seed(123)
batch_size = 2
n_sequences = 3
n_features = 5
units = 4
x_train = tf.random.normal(shape=(batch_size, n_sequences, n_features))

def get_weight(shape, dtype):
    tf.random.set_seed(123)
    return tf.Variable(tf.random.normal(shape=shape, dtype=dtype))


class customLSTM1(tf.keras.layers.Layer):
    def __init__(self, units, name="customLSTM1", **kwargs):
        super(customLSTM1, self).__init__(name=name, **kwargs)
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.Wf = self.add_weight("Wf", shape=(self.units + input_dim, self.units))
        self.bf = self.add_weight("bf", shape=(1, self.units))
        self.Wi = self.add_weight("Wi", shape=(self.units + input_dim, self.units))
        self.bi = self.add_weight("bi", shape=(1, self.units))
        self.Wo = self.add_weight("Wo", shape=(self.units + input_dim, self.units))
        self.bo = self.add_weight("bo", shape=(1, self.units))
        self.Wc = self.add_weight("Wc", shape=(self.units + input_dim, self.units))
        self.bc = self.add_weight("bc", shape=(1, self.units))
        super(customLSTM1, self).build(input_shape)

    def lstm_cell_forward(self, xt, c_prev, h_prev):
        concat_x = tf.concat([h_prev, xt], axis=1)
        forget_gate = tf.nn.sigmoid(tf.matmul(concat_x, self.Wf) + self.bf)
        input_gate = tf.nn.sigmoid(tf.matmul(concat_x, self.Wi) + self.bi)
        output_gate = tf.nn.sigmoid(tf.matmul(concat_x, self.Wo) + self.bo)
        ctt = tf.nn.tanh(tf.matmul(concat_x, self.Wc) + self.bc)
        c_next = forget_gate * c_prev + input_gate * ctt
        h_next = output_gate * tf.nn.tanh(c_next)
        return h_next, c_next, h_next

    def call(self, x):
        batch_size, n_sequences = x.shape[0], x.shape[1]
        c_prev = tf.zeros(shape=(batch_size, self.units))
        h_prev = tf.zeros(shape=(batch_size, self.units))
        for t in range(n_sequences):
            h_next, c_next, _ = self.lstm_cell_forward(x[:, t, :], c_prev, h_prev)
            h_prev = h_prev
            c_prev = c_next
        return h_next


mylstm = customLSTM1(units)
mylstm.build(input_shape=(None, n_sequences, n_features))
mylstm(x_train)


class customLSTM2(tf.keras.layers.Layer):
    def __init__(self, units,
                 name="customLSTM2",
                 use_bias=True,
                 kernel_initializer=tf.keras.initializers.glorot_normal(),
                 recurrent_initializer=tf.keras.initializers.glorot_normal(),
                 bias_initlizer=tf.keras.initializers.zeros(),
                 activation=tf.nn.tanh,
                 recurrent

                 unit_forget_bias=True, **kwargs):

        super(customLSTM2, self).__init__(name=name, **kwargs)
        self.units = units
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initlizer
        self.unit_forget_bias = unit_forget_bias

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(name="kernel", shape=(input_dim, self.units * 4), initializer=self.kernel_initializer)
        self.recurrent_kernel = self.add_weight(name="recurrent_kernel", shape=(self.units, self.units * 4), initializer=self.recurrent_initializer)
        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, dtype):
                    return tf.concat([
                        self.bias_initializer(shape=(1, self.units), dtype=dtype),
                        tf.ones(shape=(1, self.units), dtype=dtype),
                        self.bias_initializer(shape=(1, self.units * 2), dtype=dtype)
                    ], axis=1)
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(name="bias", shape=(1, self.units * 4), initializer=bias_initializer)
        else:
            self.bias = None
        super(customLSTM2, self).build(input_shape)

    def lstm_cell_forward(self, xt, c_prev, h_prev):
        z = tf.matmul(xt, self.kernel)
        z += tf.matmul(h_prev, self.recurrent_kernel)
        if self.use_bias:
            z += self.bias
        z0, z1, z2, z3 = tf.split(z, 4, axis=-1)
        input_gate = tf.nn.sigmoid(z0)
        forget_gate = tf.nn.sigmoid(z1)
        ctt = tf.nn.tanh(z2)
        output_gate = tf.nn.sigmoid(z3)
        c_next = forget_gate * c_prev + input_gate * ctt
        h_next = output_gate * tf.nn.tanh(c_next)
        return h_next, c_next, h_next

    def call(self, x):
        batch_size, n_sequences = x.shape[0], x.shape[1]
        c_prev = tf.zeros(shape=(batch_size, self.units))
        h_prev = tf.zeros(shape=(batch_size, self.units))
        for t in range(n_sequences):
            h_next, c_next, _ = self.lstm_cell_forward(x[:, t, :], c_prev, h_prev)
            h_prev = h_next
            c_prev = c_next
        return h_next


mylstm2 = customLSTM2(units, kernel_initializer=get_weight, recurrent_initializer=get_weight)
mylstm2.build(input_shape=(None, n_sequences, n_features))

lstm = tf.keras.layers.LSTM(units, kernel_initializer=get_weight, recurrent_initializer=get_weight)
mylstm2(x_train) == lstm(x_train)

# dir(tf.keras.initializers


# # In[ ]:
# #
# # def lstm_forward(x, a0, parameters):
# #     """
# #     Arguments:
# #     x -- Input data for every time-step, of shape (n_x, m, T_x).
# #     a0 -- Initial hidden state, of shape (n_a, m)
# #     parameters -- python dictionary containing:
# #                  Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
# #                  bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
# #                  Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
# #                  bi -- Bias of the update gate, numpy array of shape (n_a, 1)
# #                  Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
# #                  bc -- Bias of the first "tanh", numpy array of shape (n_a, 1)
# #                  Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
# #                  bo -- Bias of the output gate, numpy array of shape (n_a, 1)
# #                  Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
# #                  by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
# #
# #     Returns:
# #     a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
# #     y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
# #     caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
# #     """
# #     # 初始化缓存列表
# #     caches = []
# #     # 获取 x 和 参数 Wy 的维度大小
# #     n_x, m, T_x = x.shape
# #     n_y, n_a = parameters['Wy'].shape
# #     # 初始化 a, c 和 y 的值
# #     a = np.zeros((n_a, m, T_x))
# #     c = np.zeros((n_a, m, T_x))
# #     y = np.zeros((n_y, m, T_x))
# #     # 初始化 a_next 和 c_next
# #     a_next = a0
# #     c_next = np.zeros(a_next.shape)
# #     # 循环所有时间步
# #     for t in range(T_x):
# #     # 更新下一时间步隐状态值、记忆值并计算预测
# #         a_next, c_next, yt, cache = lstm_cell_forward(x[:,:,t], a_next, c_next, parameters)
# #         # 在 a 中保存新的激活值
# #         a[:,:,t] = a_next
# #         # 在 a 中保存预测值
# #         y[:,:,t] = yt
# #         # 在 c 中保存记忆值
# #         c[:,:,t]  = c_next
# #         # 添加到缓存列表
# #         caches.append(cache)
# #         # 保存各计算值供反向传播调用
# #     caches = (caches, x)
# #     return a, y, c, caches
# #
# #
# # # In[32]:
# #
# #
# # def lstm_cell_forward(xt, a_prev, c_prev, parameters):
# #     """
# #     Implement a single forward step of the LSTM-cell as described in Figure (4)
# #
# #     Arguments:
# #     xt -- your input data at timestep "t", numpy array of shape (n_x, m).
# #     a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, m)
# #     c_prev -- Memory state at timestep "t-1", numpy array of shape (n_a, m)
# #     parameters -- python dictionary containing:
# #     Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
# #     bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
# #     Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
# #     bi -- Bias of the update gate, numpy array of shape (n_a, 1)
# #     Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
# #     bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
# #     Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
# #     bo --  Bias of the output gate, numpy array of shape (n_a, 1)Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
# #     by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
# #
# #     Returns:
# #     a_next -- next hidden state, of shape (n_a, m)
# #     c_next -- next memory state, of shape (n_a, m)
# #     yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
# #     cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)
# #     """
# #
# #     # 获取参数字典中各个参数
# #     Wf = parameters["Wf"]
# #     bf = parameters["bf"]
# #     Wi = parameters["Wi"]
# #     bi = parameters["bi"]
# #     Wc = parameters["Wc"]
# #     bc = parameters["bc"]
# #     Wo = parameters["Wo"]
# #     bo = parameters["bo"]
# #     Wy = parameters["Wy"]
# #     by = parameters["by"]
# #     # 获取 xt 和 Wy 的维度参数
# #     n_x, m = xt.shape
# #     n_y, n_a = Wy.shape
# #     # 拼接 a_prev 和 xt
# #     concat = np.zeros((n_a + n_x, m))
# #     concat[: n_a, :] = a_prev
# #     concat[n_a :, :] = xt
# #     # 计算遗忘门、更新门、记忆细胞候选值、下一时间步的记忆细胞、输出门和下一时间步的隐状态值
# #     ft = sigmoid(np.matmul(Wf, concat) + bf)
# #     it = sigmoid(np.matmul(Wi, concat) + bi)
# #     cct = np.tanh(np.matmul(Wc, concat) + bc)
# #     c_next = ft*c_prev + it*cct
# #     ot = sigmoid(np.matmul(Wo, concat) + bo)
# #     a_next = ot*np.tanh(c_next)
# #     # 计算 LSTM 的预测输出
# #     yt_pred = softmax(np.matmul(Wy, a_next) + by)
# #     # 保存各计算结果值
# #     cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
# #     return a_next, c_next, yt_pred, cache
# #
# #
# # # In[ ]:
# #
# #
# # def lstm_cell_backward(da_next, dc_next, cache):
# #     """
# #     Arguments:
# #     da_next -- Gradients of next hidden state, of shape (n_a, m)
# #     dc_next -- Gradients of next cell state, of shape (n_a, m)
# #     cache -- cache storing information from the forward pass
# #
# #     Returns:
# #     gradients -- python dictionary containing:
# #      dxt -- Gradient of input data at time-step t, of shape (n_x, m)
# #      da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
# #      dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
# #      dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
# #      dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
# #      dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
# #      dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
# #      dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
# #      dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
# #      dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
# #      dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
# #     """
# #
# #     # 获取缓存值
# #     (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache    # 获取 xt 和 a_next 的维度大小
# #     n_x, m = xt.shape
# #     n_a, m = a_next.shape
# #     # 计算各种门的梯度
# #     dot = da_next * np.tanh(c_next) * ot * (1 - ot)
# #     dcct = dc_next * it + ot * (1 - np.tanh(c_next) ** 2) * it * da_next * cct * (1 - np.tanh(cct) ** 2)
# #     dit = dc_next * cct + ot * (1 - np.tanh(c_next) ** 2) * cct * da_next * it * (1 - it)
# #     dft = dc_next * c_prev + ot * (1 - np.tanh(c_next) ** 2) * c_prev * da_next * ft * (1 - ft)    # 计算各参数的梯度
# #     dWf = np.dot(dft, np.concatenate((a_prev, xt), axis=0).T)
# #     dWi = np.dot(dit, np.concatenate((a_prev, xt), axis=0).T)
# #     dWc = np.dot(dcct, np.concatenate((a_prev, xt), axis=0).T)
# #     dWo = np.dot(dot, np.concatenate((a_prev, xt), axis=0).T)
# #     dbf = np.sum(dft, axis=1, keepdims=True)
# #     dbi = np.sum(dit, axis=1, keepdims=True)
# #     dbc = np.sum(dcct, axis=1, keepdims=True)
# #     dbo = np.sum(dot, axis=1, keepdims=True)
# #
# #     da_prev = np.dot(parameters['Wf'][:,:n_a].T, dft) + np.dot(parameters['Wi'][:,:n_a].T, dit) + np.dot(parameters['Wc'][:,:n_a].T, dcct) + np.dot(parameters['Wo'][:,:n_a].T, dot)
# #     dc_prev = dc_next*ft + ot*(1-np.square(np.tanh(c_next)))*ft*da_next
# #     dxt = np.dot(parameters['Wf'][:,n_a:].T,dft)+np.dot(parameters['Wi'][:,n_a:].T,dit)+np.dot(parameters['Wc'][:,n_a:].T,dcct)+np.dot(parameters['Wo'][:,n_a:].T,dot)
# #
# #     # 将各梯度保存至字典
# #     gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
# #                    "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
# #     return gradients
# #
# #
# # # In[ ]:
# #
# #
# # def lstm_backward(da, caches):
# #     """
# #     Arguments:
# #     da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
# #     dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
# #     caches -- cache storing information from the forward pass (lstm_forward)
# #
# #     Returns:
# #     gradients -- python dictionary containing:
# #            dx -- Gradient of inputs, of shape (n_x, m, T_x)
# #            da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
# #            dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
# #            dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
# #            dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
# #            dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
# #            dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
# #            dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
# #            dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
# #            dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
# #     """
# #
# #     # 获取第一个缓存值
# #     (caches, x) = caches
# #     (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]    # 获取 da 和 x1 的形状大小
# #     n_a, m, T_x = da.shape
# #     n_x, m = x1.shape
# #     # 初始化各梯度值
# #     dx = np.zeros((n_x, m, T_x))
# #     da0 = np.zeros((n_a, m))
# #     da_prevt = np.zeros((n_a, m))
# #     dc_prevt = np.zeros((n_a, m))
# #     dWf = np.zeros((n_a, n_a+n_x))
# #     dWi = np.zeros((n_a, n_a+n_x))
# #     dWc = np.zeros((n_a, n_a+n_x))
# #     dWo = np.zeros((n_a, n_a+n_x))
# #     dbf = np.zeros((n_a, 1))
# #     dbi = np.zeros((n_a, 1))
# #     dbc = np.zeros((n_a, 1))
# #     dbo = np.zeros((n_a, 1))
# #     # 循环各时间步
# #     for t in reversed(range(T_x)):
# #         # 使用 lstm 单元反向传播计算各梯度值
# #         gradients = lstm_cell_backward(da[:, :, t] + da_prevt, dc_prevt, caches[t])
# #         # 保存各梯度值
# #         dx[:,:,t] = gradients['dxt']
# #         dWf = dWf + gradients['dWf']
# #         dWi = dWi + gradients['dWi']
# #         dWc = dWc + gradients['dWc']
# #         dWo = dWo + gradients['dWo']
# #         dbf = dbf + gradients['dbf']
# #         dbi = dbi + gradients['dbi']
# #         dbc = dbc + gradients['dbc']
# #         dbo = dbo + gradients['dbo']
# #
# #     da0 = gradients['da_prev']
# #
# #     gradients = {"dx": dx, "da0": da0, "dWf": dWf,"dbf": dbf, "dWi": dWi,"dbi": dbi,
# #     "dWc": dWc,"dbc": dbc, "dWo": dWo,"dbo": dbo}
# #     return gradients
#
#
# # In[48]:
#
# # In[64]:
#
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x + 10e-5))
#
# def softmax(x):
#     c = np.max(x, axis=1, keepdims=True)
#     exp_x = np.exp(x-c)
#     sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
#     return exp_x / (sum_exp_x + 10e-5)
#
# def lstm_cell_forward(xt, a_prev, c_prev, parameters):
#     """
#     xt shape: (n_samples, n_features)
#     a_prev shape: (n_samples, n_output)
#     c_prev shape: (n_samples, n_output)
#     """
#     # Wf, bf
#     # Wi, bi
#     # Wo, bo
#     # Wc, bc
#
#     Wf, bf = parameters['Wf'], parameters['bf']
#     Wi, bi = parameters['Wi'], parameters['bi']
#     Wo, bo = parameters['Wo'], parameters['bo']
#     Wc, bc = parameters['Wc'], parameters['bc']
#
#     concat_x  = np.c_[xt, a_prev]
#
#     ft = sigmoid(np.matmul(concat_x, Wf) + bf)
#     it = sigmoid(np.matmul(concat_x, Wi) + bi)
#     ot = sigmoid(np.matmul(concat_x, Wo) + bo)
#     ct = np.tanh(np.matmul(concat_x, Wc) + bc)
#     c_next = ft * c_prev + it * ct
#     a_next = ot * np.tanh(c_next)
#     y = softmax(a_next)
#     return a_next, c_next, y
#
# def lstm_forward(xs, n_neuron, parameters):
#     n_samples, n_sequences, n_features = xs.shape
#     a_prev = np.zeros((n_samples, n_neuron))
#     c_prev = np.zeros((n_samples, n_neuron))
#
#     y = []
#     for t in range(n_sequences):
#         a_prev, c_prev, _ = lstm_cell_forward(xs[:, t, :], a_prev, c_prev, parameters)
#         y.append(a_prev)
#     y = np.array(y)
#     return y, a_prev, c_prev
#
#
# nh = 4
# n_samples = 2
#
# def get_weights(shape):
#     np.random.seed(123)
#     return np.random.randn(*shape)
#
# def get_bias(shape):
#     return np.zeros(shape)
#
# kernel = get_weights((n_features, nh * 4))
# recurrent_kernel = get_weights((nh, nh *4))
# parameters = {
#     'Wi': np.r_[kernel[:, :4], recurrent_kernel[:, :4]],
#     'bi': get_bias((1, nh)),
#     'Wf': np.r_[kernel[:, 4:8], recurrent_kernel[:, 4:8]],
#     'bf': np.ones((1, nh)), # 初始情况，forget 设置为1，表示不遗忘。forget=0表示遗忘。
#     'Wc': np.r_[kernel[:, 8:12], recurrent_kernel[:, 8:12]],
#     'bc': get_bias((1, nh)),
#     'Wo': np.r_[kernel[:, 12:16], recurrent_kernel[:, 12:16]],
#     'bo': get_bias((1, nh)),
# }
#
#
# # In[66]:
#
# # lstm_cell_forward(x[:, 0, :], a_prev, c_prev, parameters=parameters)
#
# lstm_forward(x, 4, parameters)
#
# def tf_get_weights(shape, dtype):
#     np.random.seed(123)
#     return tf.Variable(np.random.randn(*shape), dtype=dtype)
#
# n_neuron = 4
