import os, sys
sys.path.append(os.getcwd())

import tflib as lib
import tensorflow as tf
import functools
import locale
locale.setlocale(locale.LC_ALL, '')


def nonlinearity(x):
    return tf.nn.relu(x)


def Normalize(args, name, inputs, labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    labels = None

    if ('Discriminator' in name) and args.NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name, [1, 2, 3], inputs, labels=labels, n_labels=10)
    elif ('Generator' in name) and args.NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name, [0, 2, 3], inputs, labels=labels, n_labels=10)
        else:
            return lib.ops.batchnorm.Batchnorm(name, [0, 2, 3], inputs, fused=True)
    else:
        return inputs


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def ResidualBlock(args, name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample == 'up':
        conv_1 = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(args, name + '.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name + '.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(args, name + '.N2', output, labels=labels)
    output = nonlinearity(output)
    output = conv_2(name + '.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output


def OptimizedResBlockDisc1(inputs, args):
    conv_1 = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=args.DIM_D)
    conv_2 = functools.partial(ConvMeanPool, input_dim=args.DIM_D, output_dim=args.DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=args.DIM_D, filter_size=1, he_init=False,
                             biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)
    output = nonlinearity(output)
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output


def Generator(args, n_samples, labels, noise=None, mode='default'):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = lib.ops.linear.Linear('Generator_{}.Input'.format(mode), 128, 4 * 4 * args.DIM_G, noise)
    output = tf.reshape(output, [-1, args.DIM_G, 4, 4])
    output = ResidualBlock(args, 'Generator_{}.1'.format(mode), args.DIM_G, args.DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock(args, 'Generator_{}.2'.format(mode), args.DIM_G, args.DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock(args, 'Generator_{}.3'.format(mode), args.DIM_G, args.DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize(args, 'Generator_{}.OutputN'.format(mode), output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator_{}.Output'.format(mode), args.DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, args.OUTPUT_DIM])


def Discriminator(args, inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output, args)
    output = ResidualBlock(args, 'Discriminator.2', args.DIM_D, args.DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock(args, 'Discriminator.3', args.DIM_D, args.DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock(args, 'Discriminator.4', args.DIM_D, args.DIM_D, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2, 3])
    output_wgan = lib.ops.linear.Linear('Discriminator.Output', args.DIM_D, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])

    return output_wgan
