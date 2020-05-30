import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model
import glob
from dataset import make_anime_dataset

# img_path = glob.glob("faces/*.jpg")

# batch_size = 32
# dataset, img_shape, _ = make_anime_dataset(img_path, batch_size, resize=64)


class Generator(Model):
    def __init__(self):
        super(Generator, self).__init__()
        filters = 64
        self.conv1 = layers.Conv2DTranspose(filters=filters * 7,
                                            kernel_size=4,
                                            strides=1,
                                            padding='valid',
                                            use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(filters=filters * 4,
                                            kernel_size=4,
                                            strides=2,
                                            padding='same',
                                            use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(filters=filters * 2,
                                            kernel_size=4,
                                            strides=2,
                                            padding='same',
                                            use_bias=False)
        self.bn3 = layers.BatchNormalization()
        self.conv4 = layers.Conv2DTranspose(filters=filters * 1,
                                            kernel_size=4,
                                            strides=2,
                                            padding='same',
                                            use_bias=False)
        self.bn4 = layers.BatchNormalization()
        self.conv5 = layers.Conv2DTranspose(filters=3,
                                            kernel_size=4,
                                            strides=2,
                                            padding='same',
                                            use_bias=False)

    def call(self, inputs, training=None):
        # [None, 100]
        x = inputs
        # [None, 1, 1, 100]
        x = inputs[:, tf.newaxis, tf.newaxis, :]
        x = tf.nn.relu(x)
        # [None, 4, 4, 512]
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        # [None, 8, 8, 256]
        x = tf.nn.relu(self.bn2(self.conv2(x), training=training))
        # [None, 16, 16, 128]
        x = tf.nn.relu(self.bn3(self.conv3(x), training=training))
        # [None, 32, 32, 64]
        x = tf.nn.relu(self.bn4(self.conv4(x), training=training))
        # [None, 64, 64, 3]
        x = self.conv5(x)
        x = tf.tanh(x)  # scale to [-1, 1]
        return x


class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        filters = 64

        self.conv1 = layers.Conv2D(filters=filters,
                                   kernel_size=4,
                                   strides=2,
                                   padding='valid',
                                   use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filters=filters * 2,
                                   kernel_size=4,
                                   strides=2,
                                   padding='valid',
                                   use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(filters=filters * 4,
                                   kernel_size=4,
                                   strides=2,
                                   padding='valid',
                                   use_bias=False)
        self.bn3 = layers.BatchNormalization()

        self.conv4 = layers.Conv2D(filters=filters * 8,
                                   kernel_size=3,
                                   strides=1,
                                   padding='valid',
                                   use_bias=False)
        self.bn4 = layers.BatchNormalization()

        self.conv5 = layers.Conv2D(filters=filters * 16,
                                   kernel_size=3,
                                   strides=1,
                                   padding='valid',
                                   use_bias=False)
        self.bn5 = layers.BatchNormalization()

        self.pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):

        # [None, 64, 64, 3]
        x = inputs
        # [None, 64, 64, 3] -> [None, 31, 31, 64] (64 + 2 x 0 - 4) // 2 + 1 = 31
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        # [None, 31, 31, 64] ->  [None, 14, 14, 128] (31 + 2 x 0 - 4) // 2 + 1 = 14
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        # [None, 14, 14, 128] -> [None, 6, 6, 256] (14 + 2 x 0 - 4) // 2 + 1 = 6
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        #  [None, 6, 6, 256] -> [None, 4, 4, 512] (6 + 2 x 0 - 3) // 1 + 1 = 4
        x = tf.nn.leaky_relu(self.bn4(self.conv4(x), training=training))
        # [None, 4, 4, 512] -> [None, 2, 2, 1024] (4 + 2 x 0 - 3) // 1 + 1 = 2
        x = tf.nn.leaky_relu(self.bn5(self.conv5(x), training=training))
        # [None, 2, 2, 1024] -> [None, 1024]
        x = self.pool(x)
        logits = self.fc(x)
        return logits

if __name__ == "__main__":
    generator = Generator()
    discriminator = Discriminator()
    tf.random.set_seed(12345)

    x = tf.random.normal([2, 100])
    y = generator(x)

    print(y.shape)
    print(y[0][0][0][0])  # 0.00014591764
    print(y[-1][-1][-1][-1])  # -0.00051474886

    z = discriminator(y)
    print(z.shape)
    print(z[0][0]) # 6.9550966e-05
    print(z[1][0]) # 1.8285426e-05
