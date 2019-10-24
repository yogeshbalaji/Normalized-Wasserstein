"""WGAN-GP ResNet for CIFAR-10"""

import os, sys
sys.path.append(os.getcwd())
import tflib as lib
import numpy as np
import tensorflow as tf
import locale
locale.setlocale(locale.LC_ALL, '')
from model import Generator, Discriminator


def train(args):
    with tf.Session() as session:
        if args.mode_frac == -1:
            args.mode_frac = None

        pi = lib.param('pi', np.array([1.0 / args.num_modes] * args.num_modes).astype(np.float32))
        pi_softmax = tf.nn.softmax(pi, dim=0)

        _iteration = tf.placeholder(tf.int32, shape=None)
        all_real_data_int = tf.placeholder(tf.int32, shape=[args.BATCH_SIZE, args.OUTPUT_DIM])
        all_real_data = tf.reshape(2 * ((tf.cast(all_real_data_int, tf.float32) / 256.) - .5),
                                   [args.BATCH_SIZE, args.OUTPUT_DIM])
        all_real_data += tf.random_uniform(shape=[args.BATCH_SIZE, args.OUTPUT_DIM], minval=0., maxval=1. / 128)

        fake_data = []
        for mode in range(args.num_modes):
            fake_data.append(Generator(args, args.BATCH_SIZE, None, mode=str(mode)))

        disc_costs = []
        disc_real = Discriminator(args, all_real_data, None)
        for mode in range(args.num_modes):
            if mode == 0:
                disc_fake = pi_softmax[mode] * Discriminator(args, fake_data[mode], None)
            else:
                disc_fake = disc_fake + pi_softmax[mode] * Discriminator(args, fake_data[mode], None)
        disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))

        alpha = tf.random_uniform(
            shape=[args.BATCH_SIZE, 1],
            minval=0.,
            maxval=1.
        )

        fake_data_samples = []
        for mode in range(args.num_modes):
            fake_data_samples.append(fake_data[mode][0:tf.cast(tf.ceil(pi_softmax[mode] * args.BATCH_SIZE), tf.int32)])
        fake_data_samples = tf.concat(fake_data_samples, axis=0)
        fake_data_samples = fake_data_samples[0:args.BATCH_SIZE, ::]

        differences = fake_data_samples - all_real_data
        interpolates = all_real_data + (alpha * differences)
        gradients = tf.gradients(Discriminator(args, interpolates, None), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = 10 * tf.reduce_mean((slopes - 1.) ** 2)
        disc_costs.append(gradient_penalty)

        disc_wgan = tf.add_n(disc_costs)
        disc_cost = disc_wgan
        disc_params = lib.params_with_name('Discriminator.')
        if args.alg == 'learnPI':
            pi_params = lib.params_with_name('pi')

        if args.DECAY:
            decay = tf.maximum(0., 1. - (tf.cast(_iteration, tf.float32) / args.ITERS))
        else:
            decay = 1.

        gen_cost = -tf.reduce_mean(disc_fake)
        gen_opt = tf.train.AdamOptimizer(learning_rate=args.LR * decay, beta1=0., beta2=0.9)
        disc_opt = tf.train.AdamOptimizer(learning_rate=args.LR * decay, beta1=0., beta2=0.9)
        gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
        disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
        gen_train_op = gen_opt.apply_gradients(gen_gv)
        disc_train_op = disc_opt.apply_gradients(disc_gv)
        if args.alg == 'learnPI':
            pi_train_op = tf.train.AdamOptimizer(learning_rate=20 * args.LR * decay, beta1=0.,
                                                 beta2=0.9).minimize(gen_cost, var_list=pi_params)

        # Function for generating samples
        fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
        fixed_noise_samples = []
        for mode in range(args.num_modes):
            fixed_noise_samples.append(Generator(args, 100, None, noise=fixed_noise, mode=str(mode)))

        def generate_image(frame, true_dist):
            for mode in range(args.num_modes):
                samples = session.run(fixed_noise_samples[mode])
                samples = ((samples + 1.) * (255. / 2)).astype('int32')
                lib.save_images.save_images(samples.reshape((100, 3, 32, 32)),
                                            '{}/samples_mode{}_{}.png'.format(args.log_dir, mode, frame))

        samples_100 = []
        for mode in range(args.num_modes):
            samples_100.append(Generator(args, 100, None, mode=str(mode)))

        if args.dataset == 'CIFAR':
            train_gen, dev_gen = lib.cifar10.load(args.BATCH_SIZE, args.DATA_DIR, mode_frac=args.mode_frac)
        elif args.dataset == 'CIFAR_CelebA':
            train_gen, dev_gen = lib.cifar10_celeba.load(args.BATCH_SIZE, args.CIFAR_DIR, args.CELEBA_DIR,
                                                         mode_frac=args.mode_frac)

        def inf_train_gen():
            while True:
                for images, _labels in train_gen():
                    yield images, _labels

        for name, grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
            print("{} Params:".format(name))
            total_param_count = 0
            for g, v in grads_and_vars:
                shape = v.get_shape()
                shape_str = ",".join([str(x) for x in v.get_shape()])

                param_count = 1
                for dim in shape:
                    param_count *= int(dim)
                total_param_count += param_count

                if g == None:
                    print("\t{} ({}) [no grad!]".format(v.name, shape_str))
                else:
                    print("\t{} ({})".format(v.name, shape_str))
            print("Total param count: {}".format(
                locale.format("%d", total_param_count, grouping=True)
            ))

        # Iterations begin
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1000)

        gen = inf_train_gen()

        for iteration in range(args.ITERS):
            if iteration > 0:
                if args.alg == 'learnPI':
                    _, _, pi_vals = session.run([gen_train_op, pi_train_op, pi_softmax], feed_dict={_iteration: iteration})
                else:
                    _, pi_vals = session.run([gen_train_op, pi_softmax], feed_dict={_iteration: iteration})

                pi_str = ''
                for mode in range(args.num_modes):
                    pi_str = pi_str + str(pi_vals[mode]) + ' ,'
                if iteration % 100 == 0:
                    print('Pi vals: {}'.format(pi_str))

            for i in range(args.N_CRITIC):
                _data, _labels = gen.next()
                _disc_cost, _ = session.run([disc_cost, disc_train_op],
                                            feed_dict={all_real_data_int: _data, _iteration: iteration})

            lib.plot.plot('cost', _disc_cost)

            if iteration % args.INCEPTION_FREQUENCY == args.INCEPTION_FREQUENCY - 1:
                saver.save(session, "{}/models/model_{}.ckpt".format(args.log_dir, iteration))

            # Calculate dev loss and generate samples every 100 iters
            if iteration % 100 == 99:
                dev_disc_costs = []
                for images, _labels in dev_gen():
                    _dev_disc_cost = session.run([disc_cost], feed_dict={all_real_data_int: images})
                    dev_disc_costs.append(_dev_disc_cost)
                lib.plot.plot('dev_cost', np.mean(dev_disc_costs))
                data_disp = ((_data + 1.) * (255. / 2)).astype('int32')
                lib.save_images.save_images(data_disp.reshape((-1, 3, 32, 32)), '{}/real.png'.format(args.log_dir))

                generate_image(iteration, _data)

            if (iteration < 500) or (iteration % 1000 == 999):
                lib.plot.flush(args.log_dir)

            lib.plot.tick()
