"""
File: local_models.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description: Layers and models used for the local autoencoder
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class ReflectionPadding2D(tf.keras.layers.Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        # Dims = (None, x, y, f)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        out_shape = (s[0], )
        for i in range(0, len(self.padding)):
            out_shape += (s[i + 1] + 2 * self.padding[i], )
        out_shape += (s[-1], )

        return out_shape

    def call(self, x, mask=None):
        i_pad, j_pad = self.padding
        paddings = [[0] * 2, [i_pad] * 2, [j_pad] * 2, [0] * 2]
        return tf.pad(x, paddings, 'REFLECT')

    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        config['padding'] = self.padding
        return config


class ShareConv2D(tf.keras.layers.Layer):

    def __init__(self,
                 weights,
                 padding,
                 strides,
                 dilations,
                 use_bias=False,
                 **kwargs):
        super(ShareConv2D, self).__init__(**kwargs)
        #  self.weights = weights
        self.use_bias = use_bias
        self.w = weights[0]
        if self.use_bias:
            self.b = weights[1]
        self.padding = padding.upper()
        self.strides = strides
        self.dilations = dilations

    def call(self, inputs):
        ti = tf.nn.conv2d(inputs,
                          self.w,
                          padding=self.padding,
                          strides=self.strides,
                          dilations=self.dilations)
        if self.use_bias:
            ti = ti + self.b
        return ti

    def get_config(self):
        config = super(ShareConv2D, self).get_config()
        config.update({
            'use_bias': self.use_bias,
            'padding': self.padding,
            'strides': self.strides,
            'dilations': self.dilations,
        })
        return config


def dense(tensor,
          neurons,
          name=None,
          activation='relu',
          use_bn=True,
          kreg=None):
    """Implements a dense layer with a Batchnormalization.

    Parameters
    ----------
    tensor : keras or tensorflow tensor
        input tensor
    neurons : int
        number of neurons
    activation : str, optional
        defines the activation function to be applyied at the endo of the
        layer.
    use_bn : bool, optional
        set if the layer will use Batchnormalization

    Returns
    -------
    Tensor for a Dense (+ BatchNormalization) + Activation

    """
    use_bias = not use_bn

    dense_name = name if activation is None and use_bias else None
    bn_name = name if activation is None and use_bn else None

    ti = tf.keras.layers.Dense(neurons,
                               use_bias=use_bias,
                               name=dense_name,
                               kernel_regularizer=kreg)(tensor)

    if use_bn:
        ti = tf.keras.layers.BatchNormalization(name=bn_name)(ti)

    if activation is not None:
        ti = tf.keras.layers.Activation(activation.lower(), name=name)(ti)
    return ti


def conv_bn(tensor,
            nf,
            ks,
            conv_dim=2,
            strides=1,
            padding='same',
            activation='relu',
            name=None,
            use_bn=True):
    """Implements the structure for making a 2D Conv followed with a
    Batchnormalization when use_bn is set to True, and then applying an
    activation function.

    Parameters
    ----------
    tensor : tensorflow or keras tensor
        input tensor
    nf : int
        number of feature to be used in the 3D Convolutional layer
    ks : int or tuple of ints
        kernel size used in the convolution
    strides : int or tuple of ints, optional
        The strides to be used in the 3D conv layer.
    padding : str, optional
        the padding to be used in the 3D conv layer.
    activation : str, optional
        the activation to be applied after the Batchnormalization layer if it
        is used or right after the 3D Conv layer otherwise.
    use_bn : bool, optional
        set if a Batchnormalization layer is used after the 3D conv layer.

    Returns
    -------
    A keras tensor which represents  a 3DConv (+BN) + Activation

    """
    conv_name = 'Conv%iD' % conv_dim
    use_bias = not use_bn

    cname = name if activation is None and use_bias else None
    bn_name = name if activation is None and use_bn else None

    ti = getattr(tf.keras.layers, conv_name)(nf,
                                             kernel_size=ks,
                                             strides=strides,
                                             use_bias=use_bias,
                                             name=cname,
                                             padding=padding)(tensor)
    if use_bn:
        ti = tf.keras.layers.BatchNormalization(name=bn_name)(ti)
    ti = tf.keras.layers.Activation(activation.lower(), name=name)(ti)
    return ti


def conv2D(tensor,
           nf,
           ks,
           strides=1,
           padding='same',
           activation='relu',
           name=None,
           use_bn=True):
    """Implements the structure for making a 2D Conv followed with a
    Batchnormalization when use_bn is set to True, and then applying an
    activation function.

    Parameters
    ----------
    tensor : tensorflow or keras tensor
        input tensor
    nf : int
        number of feature to be used in the 3D Convolutional layer
    ks : int or tuple of ints
        kernel size used in the convolution
    strides : int or tuple of ints, optional
        The strides to be used in the 2D conv layer.
    padding : str, optional
        the padding to be used in the 2D conv layer.
    activation : str, optional
        the activation to be applied after the Batchnormalization layer if it
        is used or right after the 2D Conv layer otherwise.
    use_bn : bool, optional
        set if a Batchnormalization layer is used after the 3D conv layer.

    Returns
    -------
    A keras tensor which represents  a 2DConv (+BN) + Activation

    """
    ti = conv_bn(tensor,
                 nf,
                 ks,
                 conv_dim=2,
                 strides=strides,
                 padding=padding,
                 activation=activation,
                 use_bn=use_bn,
                 name=name)
    return ti


def conv2DResizing(tensor,
                   nf,
                   ks,
                   size=1,
                   padding='same',
                   activation='relu',
                   use_bn=True):
    """Implements the structure for making a 2D Resizing using
    Upsampling + Reflection Padding and Convolution followed with a
    Batchnormalization when use_bn is set to True, and then applying an
    activation function..

    Parameters
    ----------
    tensor : tensorflow or keras tensor
        input tensor
    nf : int
        number of feature to be used in the 3D Transpose Convolutional layer
    ks : int or tuple of ints
        kernel size used in the convolution
    size : int or tuple of ints, optional
        The size factor to be used in the Upsampling layer
    padding : str, optional
        the padding to be used in the 3D conv layer.
    activation : str, optional
        the activation to be applied after the Batchnormalization layer if it
        is used or right after the 3D Conv layer otherwise.
    use_bn : bool, optional
        set if a Batchnormalization layer is used after the 3D conv layer.
    Returns
    -------
    A keras tensor which represents  a 3DTransposeConv (+BN) + Activation

    """
    use_bias = not use_bn
    ti = tf.keras.layers.UpSampling2D(size=size)(tensor)
    ti = ReflectionPadding2D()(ti)
    ti = tf.keras.layers.Conv2D(nf,
                                kernel_size=ks,
                                strides=1,
                                use_bias=use_bias,
                                padding=padding)(ti)
    if use_bn:
        ti = tf.keras.layers.BatchNormalization()(ti)

    ti = tf.keras.layers.Activation(activation.lower())(ti)
    return ti


def KSAC(ti, nf, rates=[4, 8], activation='swish'):
    """Original Kernel-Sharing Atrous Convolution block Huan et al.
    """
    # 1x1 Conv.
    t1 = tf.keras.layers.Conv2D(nf,
                                1,
                                strides=1,
                                padding='same',
                                use_bias=False)(ti)
    # Global Average Pool + Conv + BN + Activation + Upsampling
    t2 = tf.reduce_mean(ti, axis=(1, 2))
    t2 = t2[:, tf.newaxis, tf.newaxis, :]  # Expand dimensions (None,1,1,#)

    t2 = tf.keras.layers.Conv2D(nf,
                                1,
                                strides=1,
                                padding='same',
                                use_bias=False)(t2)

    t2 = tf.keras.layers.BatchNormalization()(t2)

    t2 = tf.keras.layers.Activation(activation.lower())(t2)
    t2 = tf.keras.layers.UpSampling2D(K.int_shape(ti)[1:3])(t2)
    # Sharing Conv. Layer
    cl = tf.keras.layers.Conv2D(nf,
                                3,
                                strides=1,
                                padding='same',
                                use_bias=False)
    # Share Dilate convolution
    t = cl(ti)  # Dilation 1 + BN + Act
    t = tf.keras.layers.BatchNormalization()(t)
    t = tf.keras.layers.Activation(activation.lower())(t)
    kst = [t]
    for r in rates:
        # 2D -> H,W
        # factor to dilate for [N,H,W,C]
        #  dilations = (1, ) + (2**1, ) * 2 + (1, )
        dilations = (1, ) + (r, ) * 2 + (1, )
        strides = (1, ) + cl.strides + (1, )
        t = ShareConv2D(cl.weights,
                        padding=cl.padding,
                        strides=strides,
                        dilations=dilations)(ti)
        t = tf.keras.layers.BatchNormalization()(t)
        t = tf.keras.layers.Activation(activation.lower())(t)
        kst.append(t)

    # Concatenation
    t_output = tf.keras.layers.Concatenate()([t1, t2] + kst)
    # Combine to fix into nf
    t_output = conv2D(t_output, nf, 1, activation=activation)
    return t_output


def ChannelAttention(f, rate, axis, mlp_activation, name=None):
    x1 = tf.reduce_max(f, axis=axis)
    x2 = tf.reduce_mean(f, axis=axis)
    c = f.shape[-1]  # Channel axis is -1
    if rate is None:
        rate = 2
    r = int(c // rate)
    if r == 0:
        r = 1
    # Create MLP
    mlp = tf.keras.models.Sequential()
    mlp.add(tf.keras.layers.Input(shape=(c, )))
    mlp.add(tf.keras.layers.Dense(r, use_bias=False))
    mlp.add(tf.keras.layers.Activation(mlp_activation.lower()))
    mlp.add(tf.keras.layers.Dense(c, use_bias=False))

    ti = mlp(x1) + mlp(x2)
    Mc = tf.keras.layers.Activation('sigmoid', name=name)(ti)
    # Expand dims
    if isinstance(axis, tuple) and len(axis) == 3:
        Mc = Mc[:, tf.newaxis, tf.newaxis, tf.newaxis]
    elif isinstance(axis, tuple) and len(axis) == 2:
        Mc = Mc[:, tf.newaxis, tf.newaxis]
    else:
        Mc = Mc[:, tf.newaxis]

    return Mc


def SpatialAttention(f1, conv_dim=2, name=None):
    x1 = tf.reduce_max(f1, axis=-1)  # Channel axis is -1
    x2 = tf.reduce_mean(f1, axis=-1)
    # Expand dims
    x1 = x1[..., tf.newaxis]  # x1 = tf.expand_dims(x1, axis=-1)
    x2 = x2[..., tf.newaxis]  # x2 = tf.expand_dims(x2, axis=-1)

    x = tf.concat([x1, x2], axis=-1)
    # Avoiding the use of Separable Convolution here
    conv_name = 'Conv%iD' % conv_dim
    Ms = getattr(tf.keras.layers, conv_name)(1,
                                             3,
                                             padding='same',
                                             activation='sigmoid',
                                             name=name)(x)
    return Ms


def CBAM2D(tensor,
           nf,
           ks,
           rate=None,
           name=None,
           strides=1,
           padding='same',
           activation='relu',
           use_bn=True):
    """ Convolutional Block Attention Module Woo et al. 2018.

    Parameters
    ----------
    tensor : tensorflow or keras tensor
        input tensor
    nf : int
        number of feature to be used in the 3D Convolutional layer
    ks : int or tuple of ints
        kernel size used in the convolution
    strides : int or tuple of ints, optional
        The strides to be used in the 3D conv layer.
    padding : str, optional
        the padding to be used in the 3D conv layer.
    activation : str, optional
        the activation to be applied after the Batchnormalization layer if it
        is used or right after the 3D Conv layer otherwise.
    use_bn : bool, optional
        set if a Batchnormalization layer is used after the 3D conv layer.

    Returns
    -------
    A keras tensor which represents  a 3DConv (+BN) + Activation

    """
    ti = conv2D(tensor,
                nf,
                ks,
                activation=activation,
                name=name,
                strides=strides,
                padding=padding,
                use_bn=use_bn)

    axis = (1, 2)
    Mc = ChannelAttention(ti, rate, axis, 'relu', name=name + '_ChAtt')
    ti = ti * Mc  # Tf makes the broadcasting for us

    Ms = SpatialAttention(ti, conv_dim=2, name=name + '_SpAtt')
    ti = ti * Ms  # TF makes the broadcast for us

    # Residual Block (if spatial input is reduced it is needed a projection)
    nf_tensor = tensor.shape[-1]
    if strides > 1 or padding.lower() != 'same' or nf_tensor != nf:
        # Projection Conv with k=1 and no activation
        # Avoiding the use of Separable Convolution here
        rb = conv2D(tensor,
                    nf,
                    1,
                    activation='linear',
                    strides=strides,
                    padding=padding,
                    use_bn=use_bn)
    else:
        rb = tensor

    ti = rb + ti

    return ti


def ae(
        input_shape,
        z_dim,
        activation='swish',
        #  dw_separable=False,
        #  verbose=True,
        suffix=''):
    """ Using Kernel Sharing conv KSC and Conv. Block Attention Module.
    """
    # Encoder
    input_enc = tf.keras.layers.Input(shape=input_shape)
    ti = KSAC(input_enc, 128, rates=[2, 4], activation=activation)
    ti = CBAM2D(ti, 128, 3, strides=2, name='CBAM1', activation=activation)
    ti = KSAC(ti, 128, rates=[2, 4], activation=activation)
    ti = CBAM2D(ti, 128, 3, strides=2, name='CBAM2', activation=activation)
    ti = CBAM2D(ti, 128, 3, activation=activation, name='CBAM3')
    ti = CBAM2D(ti, 128, 3, strides=2, name='CBAM4', activation=activation)
    ti = CBAM2D(ti, 128, 3, activation=activation, name='CBAM5')
    dim_before_flatten = np.array(ti.shape[1:])
    ti = tf.keras.layers.Flatten()(ti)
    out_enc = dense(ti, z_dim, use_bn=True, activation='linear')
    enc = tf.keras.Model(input_enc, out_enc, name='Encoder' + suffix)

    # Decoder (According to [1] decoder dose't use BN)
    input_dec = tf.keras.layers.Input(shape=z_dim)
    ti = dense(input_dec,
               dim_before_flatten.prod(),
               use_bn=True,
               activation=activation)
    ti = tf.keras.layers.Reshape(dim_before_flatten)(ti)
    ti = conv2D(ti, 128, 3, use_bn=True, padding='same', activation=activation)
    ti = conv2DResizing(ti, 128, 3, size=2, use_bn=True, activation=activation)
    ti = conv2D(ti,
                128,
                3,
                use_bn=True,
                padding='valid',
                activation=activation)
    ti = conv2DResizing(ti, 128, 3, size=2, use_bn=True, activation=activation)
    ti = conv2D(ti,
                128,
                3,
                use_bn=True,
                padding='valid',
                activation=activation)

    ti = conv2DResizing(ti, 128, 3, size=2, use_bn=True, activation=activation)

    out_dec = conv2D(ti,
                     1,
                     3,
                     padding='valid',
                     activation='linear',
                     use_bn=False)

    dec = tf.keras.Model(input_dec, out_dec, name='Decoder' + suffix)

    # AE
    input_ae = tf.keras.layers.Input(shape=input_shape)
    out_ae = dec(enc(input_ae))
    ae = tf.keras.Model(input_ae, out_ae, name='AE' + suffix)
    return ae


def dense_model(input_dim,
                latent_dim,
                nclass,
                neurons,
                nlayers=1,
                activation='relu',
                kreg=None,
                verbose=True):
    input_ti = tf.keras.layers.Input(shape=input_dim)
    ti = dense(input_ti,
               neurons,
               use_bn=True,
               kreg=kreg,
               activation=activation)
    ti = tf.keras.layers.Dropout(0.4)(ti)
    for _ in range(nlayers - 1):
        ti = dense(ti, neurons, use_bn=True, kreg=kreg, activation=activation)
        ti = tf.keras.layers.Dropout(0.4)(ti)

    z_ti = dense(
        ti,
        latent_dim,
        use_bn=True,
        kreg=kreg,
        #  activation=activation)
        activation='linear')  # Linear activation for the latent space
    ti = tf.keras.layers.Dropout(0.4)(z_ti)
    if nclass > 1:
        out_disc = dense(ti, nclass, activation='softmax', use_bn=False)
    else:
        out_disc = dense(ti, nclass, activation='sigmoid', use_bn=False)

    model = tf.keras.models.Model(input_ti, out_disc, name='Dense_Model')
    model_z = tf.keras.models.Model(input_ti, z_ti, name='Model_z')

    if verbose:
        model.summary()
        model_z.summary()
    return model_z, model
