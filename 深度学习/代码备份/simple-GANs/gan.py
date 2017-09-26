# -*- coding: UTF-8 -*-
'''
An example of distribution approximation using Generative Adversarial Networks
in TensorFlow.

Based on the blog post by Eric Jang:
http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html,

and of course the original GAN paper by Ian Goodfellow et. al.:
https://arxiv.org/abs/1406.2661.

The minibatch discrimination technique is taken from Tim Salimans et. al.:
https://arxiv.org/abs/1606.03498.
'''
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sns

sns.set(color_codes=True)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

'''
numpy.random.normal(loc=0.0, scale=1.0, size=None)
loc：float
    此概率分布的均值（对应着整个分布的中心centre）
scale：float
    此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
size：int or tuple of ints
    输出的shape，默认为None，只输出一个值
我们更经常会用到的np.random.randn(size)所谓标准正态分布（μ=0,σ=1），对应于np.random.normal(loc=0, scale=1, size)。
'''
class DataDistribution(object):
    def __init__(self):
        self.mu = 4
        self.sigma = 0.5

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()
        return samples #输出N个数据，符合正态分布


#生成器输入噪声分布,样本首先在指定范围内均匀生成，然后随机扰动
class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01

#全连接网络层 
def linear(input, output_dim, scope=None, stddev=1.0):
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable(
            'w',
            [input.get_shape()[1], output_dim],
            initializer=tf.random_normal_initializer(stddev=stddev)
        )
        b = tf.get_variable(
            'b',
            [output_dim],
            initializer=tf.constant_initializer(0.0)
        )
        return tf.matmul(input, w) + b

#生成器网络
#生成器是通过非线性（softplus函数）的线性变换，接着是另一个线性变换。
def generator(input, h_dim):
    h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h1 = linear(h0, 1, 'g1')
    return h1

#鉴别器网络
#在这种情况下，我们发现重要的是确保判别器比发生器更强大，否则它们没有足够的能力学习从而准确区分生成的和实际的样本。 
#所以我们做了一个更深层的神经网络，具有更大的维度。 它使用除了最后一个层之外的所有层中的tanh非线性，其是sigmoid（其输出可以被解释为概率）
def discriminator(input, h_dim, minibatch_layer=True):
    h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
    h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    if minibatch_layer:
        h2 = minibatch(h1)
    else:
        h2 = tf.nn.relu(linear(h1, h_dim * 2, scope='d2'))

    h3 = tf.sigmoid(linear(h2, 1, scope='d3'))
    return h3


'''
提高样本多样性
根据Tim Salimans和OpenAI的合作者最近的一篇文章，生成器崩溃到其输出非常窄的点分布的参数设置的问题是GAN的主要失败模式之一。 
幸运的是，他们还提出了一个解决方案：允许判别器同时查看多个样本，这是一种称之为“小批量判别”的技术。
在这篇文章中，小批量判别被定义为任何判别器能够查看整批样本以便确定它们是来自生成器还是实际数据的方法。 
他们还提出了一种更具体的算法，通过建模给定样品与同一批次中的所有其他样品之间的距离来工作。 
然后将这些距离与原始样品组合并通过鉴别器，因此可以选择在分类过程中使用距离测量值和样品值。

取出判别器的一些中间层的输出。
将其乘以3D张量以产生矩阵，在下面的代码中大小为num_kernels x kernel_dim
在批量中的所有样本之间计算该矩阵中的行之间的L1距离，然后应用负指数。
样本的minibatch 特征是这些取幂距离的总和。
将原始输入连接到新创建的最小匹配特征的最小匹配层（前一个判别层的输出），并将其作为输入传递给判别的下一层。
'''
def minibatch(input, num_kernels=5, kernel_dim=3):
    print(input.shape)
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    print(x.shape)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    print(activation.shape)
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    print(minibatch_features.shape)
    return tf.concat([input, minibatch_features], 1)

#优化器
def optimizer(loss, var_list):
    learning_rate = 0.001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=step,var_list=var_list)
    return optimizer


def log(x):
    '''
    Sometimes discriminiator outputs can reach values close to
    (or even slightly less than) zero due to numerical rounding.
    This just makes sure that we exclude those values so that we don't
    end up with NaNs during optimisation.
    '''
    return tf.log(tf.maximum(x, 1e-5))


class GAN(object):
    def __init__(self, params):
        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.
        with tf.variable_scope('G'):
            #z为噪音
            self.z = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
            #默认隐层为4层，将噪音z通过G
            self.G = generator(self.z, params.hidden_size)

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        #
        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.

        #x为真实的数据
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, 1))
        #鉴别器输入为真实数据或者伪造数据，并通过鉴别器网络
        with tf.variable_scope('D'):
            self.D1 = discriminator(
                self.x,
                params.hidden_size,
                params.minibatch
            )
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(
                self.G,
                params.hidden_size,
                params.minibatch
            )

        #判别模型的损失函数，交叉熵 D1为真实数据通过鉴别器的输出概率 D2为伪造数据通过鉴别器的输出概率
        #鉴别器的输出为 认为一个样本是真实的概率 
        #优化的目标是D1尽量大 D2尽量小
        self.loss_d = tf.reduce_mean(-log(self.D1) - log(1 - self.D2))
        #生成器的损失函数 D2尽量大
        self.loss_g = tf.reduce_mean(-log(self.D2))

        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)

#训练过程
def train(model, data, gen, params):
    anim_frames = []

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        #这里的训练次数默认为5000，batch-size默认为8
        for step in range(params.num_steps + 1):
            #训练鉴别器，鉴别器能力提升
            x = data.sample(params.batch_size)  #真实数据
            z = gen.sample(params.batch_size)   #伪造数据
            #计算鉴别器的损失函数，数据有真实数据和伪造数据
            loss_d, _, = session.run([model.loss_d, model.opt_d], {
                model.x: np.reshape(x, (params.batch_size, 1)),
                model.z: np.reshape(z, (params.batch_size, 1))
            })

            #训练生成器，生成器能力提升
            #继续伪造数据
            z = gen.sample(params.batch_size)
            loss_g, _ = session.run([model.loss_g, model.opt_g], {
                model.z: np.reshape(z, (params.batch_size, 1))
            })

            #打印
            if step % params.log_every == 0:
                print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))

            if params.anim_path and (step % params.anim_every == 0):
                anim_frames.append(samples(model, session, data, gen.range, params.batch_size))

        if params.anim_path:
            save_animation(anim_frames, params.anim_path, gen.range)
        else:
            samps = samples(model, session, data, gen.range, params.batch_size)
            plot_distributions(samps, gen.range)


def samples(
    model,
    session,
    data,
    sample_range,
    batch_size,
    num_points=10000,
    num_bins=100
):
    '''
    Return a tuple (db, pd, pg), where db is the current decision
    boundary, pd is a histogram of samples from the data distribution,
    and pg is a histogram of generated samples.
    '''
    xs = np.linspace(-sample_range, sample_range, num_points)
    bins = np.linspace(-sample_range, sample_range, num_bins)

    # decision boundary
    #用于评估
    #num_points//batch_size = 10000/8 = 1250 分1250组 每个点的长度为16/10000
    #D1为真实数据通过鉴别器概率
    #x经过鉴别器为输出概率D1
    #x取值为(-8,8)之间分1250组，评估x的取值在哪个区间，鉴别器的准确度最好
    db = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db[batch_size * i:batch_size * (i + 1)] = session.run(
            model.D1,
            {
                model.x: np.reshape(xs[batch_size * i:batch_size * (i + 1)],(batch_size, 1))
            }
        )

    # data distribution
    #按正态分布产生10000个点，绘制成直方图，直方图间隔为100
    d = data.sample(num_points)
    pd, _ = np.histogram(d, bins=bins, density=True)

    # generated samples
    # 噪声通过生成网络生成的数据的分布情况，并绘制成直方图
    zs = np.linspace(-sample_range, sample_range, num_points)
    g = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g[batch_size * i:batch_size * (i + 1)] = session.run(
            model.G,
            {
                model.z: np.reshape(zs[batch_size * i:batch_size * (i + 1)], (batch_size, 1))
            }
        )
    pg, _ = np.histogram(g, bins=bins, density=True)

    return db, pd, pg


def plot_distributions(samps, sample_range):
    db, pd, pg = samps
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))
    f, ax = plt.subplots(1)
    ax.plot(db_x, db, label='decision boundary')
    ax.set_ylim(0, 1)
    plt.plot(p_x, pd, label='real data')
    plt.plot(p_x, pg, label='generated data')
    plt.title('1D Generative Adversarial Network')
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    plt.legend()
    plt.show()


def save_animation(anim_frames, anim_path, sample_range):
    f, ax = plt.subplots(figsize=(6, 4))
    f.suptitle('1D Generative Adversarial Network', fontsize=15)
    plt.xlabel('Data values')
    plt.ylabel('Probability density')
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 1.4)
    line_db, = ax.plot([], [], label='decision boundary')
    line_pd, = ax.plot([], [], label='real data')
    line_pg, = ax.plot([], [], label='generated data')
    frame_number = ax.text(
        0.02,
        0.95,
        '',
        horizontalalignment='left',
        verticalalignment='top',
        transform=ax.transAxes
    )
    ax.legend()

    db, pd, _ = anim_frames[0]
    db_x = np.linspace(-sample_range, sample_range, len(db))
    p_x = np.linspace(-sample_range, sample_range, len(pd))

    def init():
        line_db.set_data([], [])
        line_pd.set_data([], [])
        line_pg.set_data([], [])
        frame_number.set_text('')
        return (line_db, line_pd, line_pg, frame_number)

    def animate(i):
        frame_number.set_text(
            'Frame: {}/{}'.format(i, len(anim_frames))
        )
        db, pd, pg = anim_frames[i]
        line_db.set_data(db_x, db)
        line_pd.set_data(p_x, pd)
        line_pg.set_data(p_x, pg)
        return (line_db, line_pd, line_pg, frame_number)

    anim = animation.FuncAnimation(
        f,
        animate,
        init_func=init,
        frames=len(anim_frames),
        blit=True
    )
    anim.save(anim_path, fps=30, extra_args=['-vcodec', 'libx264'])


def main(args):
    model = GAN(args)
    train(model, DataDistribution(), GeneratorDistribution(range=8), args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=5000,
                        help='the number of training steps to take')
    parser.add_argument('--hidden-size', type=int, default=4,
                        help='MLP hidden size')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='the batch size')
    parser.add_argument('--minibatch', action='store_true',
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    parser.add_argument('--anim-path', type=str, default=None,
                        help='path to the output animation file')
    parser.add_argument('--anim-every', type=int, default=100,
                        help='save every Nth frame for animation')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
