[toc]

# Tensorflow 和 Numpy 函数对比

| 功能 | tensorflow | numpy |
| -- | -- | -- |
| sum | tf.reduce_sum(x) | np.sum(x) or x.sum() | 
| 转置 | tf.traspose(x) | x.T | 
| 数据类型转换 | tf.cast(x, tf.float32) | x.astype(np.float32) | 
| | tf.squeeze(x) | np.squeeze(x) or x.squeeze() |
| 设置随机数种子 | tf.random.set_seed(123) |  np.random.seed(123) | 
| 生成正态分布随机数 | tf.random.normal(shape=(3, 4)) | np.random.normal(size=(3,4)) |
| | tf.reshape(x, (3,4)) | np.shape(x, (3,4) or x.shape(3, 4))
