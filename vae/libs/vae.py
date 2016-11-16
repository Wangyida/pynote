"""Convolutional/Variational autoencoder, including demonstration of
training such a network on MNIST, CelebNet and the film, "Sita Sings The Blues"
using an image pipeline.

Copyright Parag K. Mital, January 2016
"""
import tensorflow as tf
import numpy as np
import os
from libs.dataset_utils import create_input_pipeline
from libs.datasets import CELEB, MNIST
from libs.batch_norm import batch_norm
from libs import utils


def VAE(input_shape=[None, 784],
        n_filters=[64, 64, 64],
        filter_sizes=[4, 4, 4],
        n_hidden=32,
        n_code=2,
        activation=tf.nn.tanh,
        dropout=False,
        denoising=False,
        convolutional=False,
        variational=False):
    """(Variational) (Convolutional) (Denoising) Autoencoder.

    Uses tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Shape of the input to the network. e.g. for MNIST: [None, 784].
    n_filters : list, optional
        Number of filters for each layer.
        If convolutional=True, this refers to the total number of output
        filters to create for each layer, with each layer's number of output
        filters as a list.
        If convolutional=False, then this refers to the total number of neurons
        for each layer in a fully connected network.
    filter_sizes : list, optional
        Only applied when convolutional=True.  This refers to the ksize (height
        and width) of each convolutional layer.
    n_hidden : int, optional
        Only applied when variational=True.  This refers to the first fully
        connected layer prior to the variational embedding, directly after
        the encoding.  After the variational embedding, another fully connected
        layer is created with the same size prior to decoding.  Set to 0 to
        not use an additional hidden layer.
    n_code : int, optional
        Only applied when variational=True.  This refers to the number of
        latent Gaussians to sample for creating the inner most encoding.
    activation : function, optional
        Activation function to apply to each layer, e.g. tf.nn.relu
    dropout : bool, optional
        Whether or not to apply dropout.  If using dropout, you must feed a
        value for 'keep_prob', as returned in the dictionary.  1.0 means no
        dropout is used.  0.0 means every connection is dropped.  Sensible
        values are between 0.5-0.8.
    denoising : bool, optional
        Whether or not to apply denoising.  If using denoising, you must feed a
        value for 'corrupt_prob', as returned in the dictionary.  1.0 means no
        corruption is used.  0.0 means every feature is corrupted.  Sensible
        values are between 0.5-0.8.
    convolutional : bool, optional
        Whether or not to use a convolutional network or else a fully connected
        network will be created.  This effects the n_filters parameter's
        meaning.
    variational : bool, optional
        Whether or not to create a variational embedding layer.  This will
        create a fully connected layer after the encoding, if `n_hidden` is
        greater than 0, then will create a multivariate gaussian sampling
        layer, then another fully connected layer.  The size of the fully
        connected layers are determined by `n_hidden`, and the size of the
        sampling layer is determined by `n_code`.

    Returns
    -------
    model : dict
        {
            'cost': Tensor to optimize.
            'Ws': All weights of the encoder.
            'x': Input Placeholder
            'z': Inner most encoding Tensor (latent features)
            'y': Reconstruction of the Decoder
            'keep_prob': Amount to keep when using Dropout
            'corrupt_prob': Amount to corrupt when using Denoising
            'train': Set to True when training/Applies to Batch Normalization.
        }
    """
    # network input / placeholders for train (bn) and dropout
    x = tf.placeholder(tf.float32, input_shape, 'x')
    x_obj = tf.placeholder(tf.float32, input_shape, 'x_obj')
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    corrupt_prob = tf.placeholder(tf.float32, [1])

    if denoising:
        current_input = utils.corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)

    # 2d -> 4d if convolution
    x_tensor = utils.to_tensor(x) if convolutional else x
    current_input = x_tensor

    Ws = []
    shapes = []

    # Build the encoder
    for layer_i, n_output in enumerate(n_filters):
        with tf.variable_scope('encoder/{}'.format(layer_i)):
            shapes.append(current_input.get_shape().as_list())
            if convolutional:
                h, W = utils.conv2d(x=current_input,
                                    n_output=n_output,
                                    k_h=filter_sizes[layer_i],
                                    k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input,
                                    n_output=n_output)
            h = activation(batch_norm(h, phase_train, 'bn' + str(layer_i)))
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            Ws.append(W)
            current_input = h

    shapes.append(current_input.get_shape().as_list())

    with tf.variable_scope('variational'):
        if variational:
            dims = current_input.get_shape().as_list()
            flattened = utils.flatten(current_input)

            if n_hidden:
                h = utils.linear(flattened, n_hidden, name='W_fc')[0]
                h = activation(batch_norm(h, phase_train, 'fc/bn'))
                if dropout:
                    h = tf.nn.dropout(h, keep_prob)
            else:
                h = flattened

            z_mu = utils.linear(h, n_code, name='mu')[0]
            z_log_sigma = 0.5 * utils.linear(h, n_code, name='log_sigma')[0]
            # modified by yidawang
            # s, u, v = tf.svd(z_log_sigma)
            # z_log_sigma = tf.matmul(tf.matmul(u, tf.diag(s)), tf.transpose(v))
            # end yidawang

            # Sample from noise distribution p(eps) ~ N(0, 1)
            epsilon = tf.random_normal(
                tf.pack([tf.shape(x)[0], n_code]))

            # Sample from posterior
            z = z_mu + tf.mul(epsilon, tf.exp(z_log_sigma))

            if n_hidden:
                h = utils.linear(z, n_hidden, name='fc_t')[0]
                h = activation(batch_norm(h, phase_train, 'fc_t/bn'))
                if dropout:
                    h = tf.nn.dropout(h, keep_prob)
            else:
                h = z

            size = dims[1] * dims[2] * dims[3] if convolutional else dims[1]
            h = utils.linear(h, size, name='fc_t2')[0]
            current_input = activation(batch_norm(h, phase_train, 'fc_t2/bn'))
            if dropout:
                current_input = tf.nn.dropout(current_input, keep_prob)

            if convolutional:
                current_input = tf.reshape(
                    current_input, tf.pack([
                        tf.shape(current_input)[0],
                        dims[1],
                        dims[2],
                        dims[3]]))
        else:
            z = current_input

    shapes.reverse()
    n_filters.reverse()
    Ws.reverse()

    n_filters += [input_shape[-1]]

    # %%
    # Decoding layers
    for layer_i, n_output in enumerate(n_filters[1:]):
        with tf.variable_scope('decoder/{}'.format(layer_i)):
            shape = shapes[layer_i + 1]
            if convolutional:
                h, W = utils.deconv2d(x=current_input,
                                      n_output_h=shape[1],
                                      n_output_w=shape[2],
                                      n_output_ch=shape[3],
                                      n_input_ch=shapes[layer_i][3],
                                      k_h=filter_sizes[layer_i],
                                      k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input,
                                    n_output=n_output)
            h = activation(batch_norm(h, phase_train, 'dec/bn' + str(layer_i)))
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            current_input = h

    y = current_input
    x_obj_flat = utils.flatten(x_obj)
    y_flat = utils.flatten(y)

    # l2 loss
    loss_x = tf.reduce_sum(tf.squared_difference(x_obj_flat, y_flat), 1)

    if variational:
        # variational lower bound, kl-divergence
        loss_z = -0.5 * tf.reduce_sum(
            1.0 + 2.0 * z_log_sigma -
            tf.square(z_mu) - tf.exp(2.0 * z_log_sigma), 1)

        # add l2 loss
        cost = tf.reduce_mean(loss_x + loss_z)
    else:
        # just optimize l2 loss
        cost = tf.reduce_mean(loss_x)

    return {'cost': cost, 'Ws': Ws,
            'x': x, 'x_obj': x_obj, 'z': z, 'y': y,
            'keep_prob': keep_prob,
            'corrupt_prob': corrupt_prob,
            'train': phase_train}


def train_vae(files_img,
              files_obj,
              input_shape,
              learning_rate=0.0001,
              batch_size=100,
              n_epochs=50,
              n_examples=10,
              crop_shape=[64, 64, 3],
              crop_factor=0.8,
              n_filters=[100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              convolutional=True,
              variational=True,
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=100,
              save_step=100,
              ckpt_name="./vae.ckpt"):
    """General purpose training of a (Variational) (Convolutional) Autoencoder.

    Supply a list of file paths to images, and this will do everything else.

    Parameters
    ----------
    files : list of strings
        List of paths to images.
    input_shape : list
        Must define what the input image's shape is.
    learning_rate : float, optional
        Learning rate.
    batch_size : int, optional
        Batch size.
    n_epochs : int, optional
        Number of epochs.
    n_examples : int, optional
        Number of example to use while demonstrating the current training
        iteration's reconstruction.  Creates a square montage, so make
        sure int(sqrt(n_examples))**2 = n_examples, e.g. 16, 25, 36, ... 100.
    crop_shape : list, optional
        Size to centrally crop the image to.
    crop_factor : float, optional
        Resize factor to apply before cropping.
    n_filters : list, optional
        Same as VAE's n_filters.
    n_hidden : int, optional
        Same as VAE's n_hidden.
    n_code : int, optional
        Same as VAE's n_code.
    convolutional : bool, optional
        Use convolution or not.
    variational : bool, optional
        Use variational layer or not.
    filter_sizes : list, optional
        Same as VAE's filter_sizes.
    dropout : bool, optional
        Use dropout or not
    keep_prob : float, optional
        Percent of keep for dropout.
    activation : function, optional
        Which activation function to use.
    img_step : int, optional
        How often to save training images showing the manifold and
        reconstruction.
    save_step : int, optional
        How often to save checkpoints.
    ckpt_name : str, optional
        Checkpoints will be named as this, e.g. 'model.ckpt'
    """
    tf.set_random_seed(1)
    seed=2
    batch_obj = create_input_pipeline(
        files=files_obj,
        batch_size=batch_size,
        n_epochs=n_epochs,
        crop_shape=crop_shape,
        crop_factor=crop_factor,
        shape=input_shape,
        seed=seed)

    batch_img = create_input_pipeline(
        files=files_img,
        batch_size=batch_size,
        n_epochs=n_epochs,
        crop_shape=crop_shape,
        crop_factor=crop_factor,
        shape=input_shape,
        seed=seed)


    ae = VAE(input_shape=[None] + crop_shape,
             convolutional=convolutional,
             variational=variational,
             n_filters=n_filters,
             n_hidden=n_hidden,
             n_code=n_code,
             dropout=dropout,
             filter_sizes=filter_sizes,
             activation=activation)

    # Create a manifold of our inner most layer to show
    # example reconstructions.  This is one way to see
    # what the "embedding" or "latent space" of the encoder
    # is capable of encoding, though note that this is just
    # a random hyperplane within the latent space, and does not
    # encompass all possible embeddings.
    zs = np.random.uniform(
        -1.0, 1.0, [4, n_code]).astype(np.float32)
    zs = utils.make_latent_manifold(zs, n_examples)

    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(ae['cost'])

    # We create a session to use the graph
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    # This will handle our threaded image pipeline
    coord = tf.train.Coordinator()

    # Ensure no more changes to graph
    tf.get_default_graph().finalize()

    # Start up the queues for handling the image pipeline
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    if os.path.exists(ckpt_name):
        saver.restore(sess, ckpt_name)

    # Fit all training data
    t_i = 0
    batch_i = 0
    epoch_i = 0
    cost = 0
    n_files = len(files_img)
    test_xs = sess.run(batch_img) / 255.0
    utils.montage(test_xs, 'test_xs_img.png')
    test_xs = sess.run(batch_obj) / 255.0
    utils.montage(test_xs, 'test_xs_obj.png')
    try:
        while not coord.should_stop():
            batch_i += 1
            batch_xs_img = sess.run(batch_img) / 255.0
            batch_xs_obj = sess.run(batch_obj) / 255.0
            train_cost = sess.run([ae['cost'], optimizer], feed_dict={
                ae['x']: batch_xs_img, ae['x_obj']: batch_xs_obj, ae['train']: True,
                ae['keep_prob']: keep_prob})[0]
            cost += train_cost
            if batch_i % n_files == 0:
                batch_i = 0
                epoch_i += 1

            if batch_i % img_step == 0:
                # Plot example reconstructions from latent layer
                recon = sess.run(
                    ae['y'], feed_dict={
                        ae['z']: zs,
                        ae['train']: False,
                        ae['keep_prob']: 1.0})
                utils.montage(recon.reshape([-1] + crop_shape),
                              'manifold_%08d.png' % t_i)

                # Plot example reconstructions
                recon = sess.run(
                    ae['y'], feed_dict={ae['x']: test_xs,
                                        ae['train']: False,
                                        ae['keep_prob']: 1.0})
                utils.montage(recon.reshape([-1] + crop_shape),
                              'reconstruction_%08d.png' % t_i)
                t_i += 1

            if batch_i % save_step == 0:
                print(train_cost)
                # Save the variables to disk.
                saver.save(sess, "./" + ckpt_name,
                           global_step=batch_i,
                           write_meta_graph=False)
    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        # One of the threads has issued an exception.  So let's tell all the
        # threads to shutdown.
        coord.request_stop()

    # Wait until all threads have finished.
    coord.join(threads)

    # Clean up the session.
    sess.close()

if __name__ == '__main__':
    test_celeb()
