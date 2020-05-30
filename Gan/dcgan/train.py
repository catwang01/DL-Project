import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import glob
from model import Generator, Discriminator
from dataset import make_anime_dataset

def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]),
                                        axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)


def celoss_ones(logits):
    # 计算属于与标签为1的交叉熵
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # 计算属于与标签为0的交叉熵
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)


def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 计算判别器的误差函数
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 判定生成图片
    d_fake_logits = discriminator(fake_image, is_training)
    # 判定真实图片
    d_real_logits = discriminator(batch_x, is_training)
    # 真实图片与1之间的误差
    d_loss_real = celoss_ones(d_real_logits)
    # 生成图片与0之间的误差
    d_loss_fake = celoss_zeros(d_fake_logits)
    # 合并误差
    loss = d_loss_fake + d_loss_real

    return loss

def g_loss_fn(generator, discriminator, batch_z, is_training):
    # 采样生成图片
    fake_image = generator(batch_z, is_training)
    # 在训练生成网络时，需要迫使生成图片判定为真
    d_fake_logits = discriminator(fake_image, is_training)
    # 计算生成图片与1之间的误差
    loss = celoss_ones(d_fake_logits)

    return loss

def main():
    tf.random.set_seed(3333)
    np.random.seed(3333)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')
    z_dim = 100  # 隐藏向量z的长度
    epochs = 1 # 训练步数
    batch_size = 64  # batch size
    learning_rate = 0.0002
    is_training = True
    k = 5

    # Path
    root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root, 'models')
    generator_ckpt_path = os.path.join(model_path, 'generator', 'generator.ckpt')
    discriminator_ckpt_path = os.path.join(model_path, 'discriminator', 'discriminator.ckpt')
    save_image_path = os.path.join(root, 'gan_images')

    # 获取数据集路径
    img_path = glob.glob('faces/*.jpg')
    print('images num:', len(img_path))

    # 构建数据集对象
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size, resize=64)
    print(dataset, img_shape)

    sample = next(iter(dataset))  # 采样
    print(f"batch_shape: {sample.shape} max: {tf.reduce_max(sample).numpy()} min: {tf.reduce_min(sample).numpy()}")

    dataset = dataset.repeat(100)  # 重复循环
    db_iter = iter(dataset)

    generator = Generator()  # 创建生成器
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()  # 创建判别器
    discriminator.build(input_shape=(None, 64, 64, 3))

    # 分别为生成器和判别器创建优化器
    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    if os.path.exists(generator_ckpt_path + '.index'):
        generator.load_weights(generator_ckpt_path)
        print('Loaded generator ckpt!!')
    if os.path.exists(discriminator_ckpt_path + '.index'):
        discriminator.load_weights(discriminator_ckpt_path)
        print('Loaded discriminator ckpt!!')

    d_losses, g_losses = [], []
    for epoch in range(epochs):  # 训练epochs次
        time_start = datetime.datetime.now()
        # 1. 训练判别器，训练k步后训练 generator
        for _ in range(k):
            # 采样隐藏向量
            batch_z = tf.random.normal([batch_size, z_dim])
            batch_x = next(db_iter)  # 采样真实图片
            # 判别器前向计算
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x,
                                   is_training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(
                zip(grads, discriminator.trainable_variables))

        # 2. 训练生成器
        # 采样隐藏向量
        batch_z = tf.random.normal([batch_size, z_dim])
        # 生成器前向计算
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        # Epoch: 0/1 TimeUsed: 0:00:10.126834 d-loss: 1.45619345 g-loss: 0.63321948
        print(f"Epoch: {epoch}/{epochs} TimeUsed: {datetime.datetime.now()-time_start} d-loss: {d_loss:.8f} g-loss: {g_loss:.8f}")

        if epoch % 100 == 0:
            z = tf.random.normal([100, z_dim]) # 可视化
            fake_image = generator(z, training=False)
            img_path = os.path.join(save_image_path, 'gan-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')

            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))

            generator.save_weights(generator_ckpt_path)
            discriminator.save_weights(discriminator_ckpt_path)


if __name__ == '__main__':
    main()
