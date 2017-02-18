# Adaptive Noise Assisted Conjugating Generative Model
TensorFlow implementation for Adaptive noise assisted Conjugating Generative Model based on Conditional Variational Autoencoder, this project is implemented based on VAE code pool of Parag K. Mital from Kadenze course on Tensorflow and modified by Yida Wang for the paper of 'Conjugating Generative Model for Object Recognition Based on 3D Models'.

Copyright reserved for Yida Wang from BUPT.

## Figures in the paper
### Method Pipeline

This is the basic pipeline for AD-CGM
![Pipeline](https://github.com/wangyida/pynote/tree/master/vae/images/pipeline_tip.eps)

### Target

There are two set of rendered images from 3D models, here we use *VTK* for rendering.
![](https://github.com/wangyida/pynote/tree/master/vae/images/target.jpg)

### Results on ShapeNet

![Input](https://github.com/wangyida/pynote/tree/master/vae/images/test_shapenet_img.jpg) ![Output](https://github.com/wangyida/pynote/tree/master/vae/images/ADCGM_shapenet_test.jpg)

### Results on ImageNet

![Input](https://github.com/wangyida/pynote/tree/master/vae/images/test_imagenet_img.jpg) ![Output](https://github.com/wangyida/pynote/tree/master/vae/images/ADCGM_imagenet_test.jpg)

### Data Distribution

![ShapeNet](https://github.com/wangyida/pynote/tree/master/vae/images/lowdim_ADCGM_shapenet.jpg) ![ImageNet](https://github.com/wangyida/pynote/tree/master/vae/images/lowdim_ADCGM_shapenet.jpg)


## Author info

### Basic info

Yida Wang

Email: yidawang.cn@gmail.com

### Publications

[ZigzagNet: Efficient Deep Learning for Real Object Recognition Based on 3D Models](https://www.researchgate.net/profile/Yida_Wang/publications?sorting=recentlyAdded)

[Self-restraint Object Recognition by Model Based CNN Learning](http://ieeexplore.ieee.org/document/7532438/)

[Face Recognition Using Local PCA Filters](http://link.springer.com/chapter/10.1007%2F978-3-319-25417-3_5)

[CNTK on Mac: 2D Object Restoration and Recognition Based on 3D Model](https://www.microsoft.com/en-us/research/academic-program/microsoft-open-source-challenge/)

[Large-Scale 3D Shape Retrieval from ShapeNet Core55](https://shapenet.cs.stanford.edu/shrec16/shrec16shapenet.pdf)

### Extracurriculars

Contributor for OpenCV and [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn);

Google Summer of Codes successor from 2015 to 2016

### Personal Links

[ResearchGate](https://www.researchgate.net/profile/Yida_Wang), [Github](https://github.com/wangyida), [GSoC 2016](https://summerofcode.withgoogle.com/archive/2016/projects/4623962327744512/), [GSoC 2015](https://www.google-melange.com/archive/gsoc/2015/orgs/opencv/projects/wangyida.html)

***

## Additional Libs
There is also implementation of ZigzagNet and SqueezeNet for compact deep learning for classification.

## Codes Explanation

network input / placeholders for train (bn) and dropout
```python
    x_img = tf.placeholder(tf.float32, input_shape, 'x_img')
    x_obj = tf.placeholder(tf.float32, input_shape, 'x_obj')
```

Input of the reconstruction network
```python
    current_input1 = utils.corrupt(x_img)*corrupt_rec + x_img*(1-corrupt_rec) \
                if (denoising and phase_train is not None) else x_img
    current_input1.set_shape(x_img.get_shape())
    # 2d -> 4d if convolution
    current_input1 = utils.to_tensor(current_input1) \
                if convolutional else current_input1
```

Encoder
```python
    for layer_i, n_output in enumerate(n_filters):
        with tf.variable_scope('encoder/{}'.format(layer_i)):
            shapes.append(current_input1.get_shape().as_list())
            if convolutional:
                h, W = utils.conv2d(x=current_input1,
                                    n_output=n_output,
                                    k_h=filter_sizes[layer_i],
                                    k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input1,
                                    n_output=n_output)
            h = activation(batch_norm(h, phase_train, 'bn' + str(layer_i)))
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            Ws.append(W)
            current_input1 = h

    shapes.append(current_input1.get_shape().as_list())

    with tf.variable_scope('variational'):
        if variational:
            dims = current_input1.get_shape().as_list()
            flattened = utils.flatten(current_input1)

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
                tf.stack([tf.shape(x_img)[0], n_code]))

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
            current_input1 = activation(batch_norm(h, phase_train, 'fc_t2/bn'))
            if dropout:
                current_input1 = tf.nn.dropout(current_input1, keep_prob)

            if convolutional:
                current_input1 = tf.reshape(
                    current_input1, tf.stack([
                        tf.shape(current_input1)[0],
                        dims[1],
                        dims[2],
                        dims[3]]))
        else:
            z = current_input1
```

Decoder
```python
    for layer_i, n_output in enumerate(n_filters[1:]):
        with tf.variable_scope('decoder/{}'.format(layer_i)):
            shape = shapes[layer_i + 1]
            if convolutional:
                h, W = utils.deconv2d(x=current_input1,
                                      n_output_h=shape[1],
                                      n_output_w=shape[2],
                                      n_output_ch=shape[3],
                                      n_input_ch=shapes[layer_i][3],
                                      k_h=filter_sizes[layer_i],
                                      k_w=filter_sizes[layer_i])
            else:
                h, W = utils.linear(x=current_input1,
                                    n_output=n_output)
            h = activation(batch_norm(h, phase_train, 'dec/bn' + str(layer_i)))
            if dropout:
                h = tf.nn.dropout(h, keep_prob)
            current_input1 = h
```

Loss finctions of VAE and softmax
```python
    # l2 loss
    loss_x = tf.reduce_mean(
        tf.reduce_sum(tf.squared_difference(x_obj_flat, y_flat), 1))
    loss_z = 0

    if variational:
        # Variational lower bound, kl-divergence
        loss_z = tf.reduce_mean(-0.5 * tf.reduce_sum(
            1.0 + 2.0 * z_log_sigma -
            tf.square(z_mu) - tf.exp(2.0 * z_log_sigma), 1))

        # Add l2 loss
        cost_vae = tf.reduce_mean(loss_x + loss_z)
    else:
        # Just optimize l2 loss
        cost_vae = tf.reduce_mean(loss_x)

    # Alexnet for clasification based on softmax using TensorFlow slim
    if softmax:
        axis = list(range(len(x_img.get_shape())))
        mean1, variance1 = tf.nn.moments(x_obj, axis) \
                        if (phase_train is True) else tf.nn.moments(x_img, axis)
        mean2, variance2 = tf.nn.moments(y, axis)
        var_prob = variance2/variance1

        # Input of the classification network
        current_input2 = utils.corrupt(x_img)*corrupt_cls + \
                     x_img*(1-corrupt_cls) \
                     if (denoising and phase_train is True) else x_img
        current_input2.set_shape(x_img.get_shape())
        current_input2 = utils.to_tensor(current_input2) \
                    if convolutional else current_input2

        y_concat = tf.concat_v2([current_input2, y], 3)
        with tf.variable_scope('deconv/concat'):
            shape = shapes[layer_i + 1]
            if convolutional:
        # Here we set the input of classification network is the twice of
        # the input of the reconstruction network
        # 112->224 for alexNet and 150->300 for inception v3 and v4
                y_concat, W = utils.deconv2d(x=y_concat,
                                          n_output_h=y_concat.get_shape()[1]*2,
                                          n_output_w=y_concat.get_shape()[1]*2,
                                          n_output_ch=y_concat.get_shape()[3],
                                          n_input_ch=y_concat.get_shape()[3],
                                          k_h=3,
                                          k_w=3)
                Ws.append(W)

        # SqueezeNet
        if classifier == 'squeezenet':
            predictions, net = squeezenet.squeezenet(
                        y_concat, num_classes=13)
        if classifier == 'zigzagnet':
            predictions, net = squeezenet.zigzagnet(
                        y_concat, num_classes=13)
        elif classifier == 'alexnet_v2':
            predictions, end_points = alexnet.alexnet_v2(
                        y_concat, num_classes=13)
        elif classifier == 'inception_v1':
            predictions, end_points = inception.inception_v1(
                        y_concat, num_classes=13)
        elif classifier == 'inception_v2':
            predictions, end_points = inception.inception_v2(
                        y_concat, num_classes=13)
        elif classifier == 'inception_v3':
            predictions, end_points = inception.inception_v3(
                        y_concat, num_classes=13)

        x_label_onehot = tf.squeeze(tf.one_hot(x_label, 13, 1, 0), [1])
        slim.losses.softmax_cross_entropy(predictions, x_label_onehot)
        cost_s = slim.losses.get_total_loss()
```

Main API for training and testing
```python
def train_vae(files_img,
              files_obj,
              input_shape,
              use_csv=False,
              learning_rate=0.0001,
              batch_size=100,
              n_epochs=50,
              n_examples=121,
              crop_shape=[128, 128, 3],
              crop_factor=0.8,
              n_filters=[75, 100, 100, 100, 100],
              n_hidden=256,
              n_code=50,
              denoising=True,
              convolutional=True,
              variational=True,
              softmax=False,
              classifier='alexnet_v2',
              filter_sizes=[3, 3, 3, 3],
              dropout=True,
              keep_prob=0.8,
              activation=tf.nn.relu,
              img_step=2500,
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
```
