"""
File: lobte.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description: This module includes all the classes used to create the propose
LobTe model
"""
import numpy as np
import tensorflow as tf

# Clear all previously registered custom objects
tf.keras.saving.get_custom_objects().clear()


def positional_encoding(length, depth):
    ndepth = np.ceil(depth / 2)

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(ndepth)[np.newaxis, :] / ndepth  # (1, ndepth)

    angle_rates = 1 / (10000**depths)  # (1, ndepth)
    angle_rads = positions * angle_rates  # (pos, ndepth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    pos_encoding = pos_encoding[:, :depth]

    return tf.cast(pos_encoding, dtype=tf.float32)


@tf.keras.saving.register_keras_serializable()
class BaseAttention(tf.keras.layers.Layer):

    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


@tf.keras.saving.register_keras_serializable()
class GlobalSelfAttention(BaseAttention):

    def __init__(self, return_attention_scores=False, **kwargs):
        super().__init__(**kwargs)
        self.return_attention_scores = return_attention_scores

    def call(self, x):
        if self.return_attention_scores:
            attn_output, attn_scores = self.mha(query=x,
                                                value=x,
                                                key=x,
                                                return_attention_scores=True)
            # Cache the attention scores for plotting later.
            self.last_attn_scores = attn_scores
        else:
            attn_output = self.mha(query=x, value=x, key=x)
            self.last_attn_scores = None
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

    def get_config(self):
        config = super(GlobalSelfAttention, self).get_config()
        config['return_attention_scores'] = self.return_attention_scores
        return config


@tf.keras.saving.register_keras_serializable()
class FeedForward(tf.keras.layers.Layer):

    def __init__(self, d_model, dff, activation, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation=activation),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


@tf.keras.saving.register_keras_serializable()
class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self,
                 d_model,
                 num_heads,
                 dff,
                 activation='relu',
                 dropout_rate=0.1,
                 return_attention_scores=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate,
            return_attention_scores=return_attention_scores)

        self.ffn = FeedForward(d_model, dff, activation)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.self_attention.last_attn_scores
        return x


@tf.keras.saving.register_keras_serializable()
class PositionalProjection(tf.keras.layers.Layer):

    def __init__(self, d_model, d_dp_rep, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_dp_rep = d_dp_rep
        # Deep-phenotype representation. This is the extra token learned in ViT
        self.dp_rep = tf.keras.initializers.GlorotUniform()(shape=(1, d_dp_rep,
                                                                   d_model))
        # Equivalent to layers.Dense(d_model, use_bias=True)
        self.linear_projection = tf.keras.layers.EinsumDense(
            'ijk,kl->ijl', output_shape=(None, d_model), bias_axes='l')
        # length should be grater that number of patches.
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def call(self, x):
        # x will be in (Batch, N patches, sequense length) = (B,N,S)
        # We perform the linear projection
        x = self.linear_projection(x)
        # NOTE: Do no use "a,b,c = x.shape" or "a,b,c=tf.shape(x)" to use the
        # axis 0, instead you should use a = tf.shape(x)
        # x will be in (Batch, N patches, model dim) = (B,N,M)
        #  B, N, M = tf.shape(x)
        B = tf.shape(x)[0]
        # Concatenate the extra token which will be used for regresion
        x0 = tf.repeat(self.dp_rep, repeats=B, axis=0)
        x = tf.concat([x0, x], axis=1)
        # Add the extra token to the N patches dimension
        x = x + self.pos_encoding[tf.newaxis, :tf.shape(x)[1], :]
        # x will be (B, N+d_dp_rep, M)
        return x

    def get_config(self):
        config = super(PositionalProjection, self).get_config()
        config['d_model'] = self.d_model
        config['d_dp_rep'] = self.d_dp_rep
        return config


@tf.keras.saving.register_keras_serializable()
class DeepPhenotypeLobeEncoder(tf.keras.layers.Layer):

    def __init__(self,
                 num_layers,
                 d_model,
                 d_dp_rep,
                 num_heads,
                 dff,
                 dropout_rate=0.25,
                 **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.d_dp_rep = d_dp_rep
        self.num_layers = num_layers

        self.pos_projection = PositionalProjection(d_model, d_dp_rep)

        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         activation='relu',
                         dropout_rate=dropout_rate)
            for _ in range(num_layers - 1)
        ]
        # Return attentions scores for the last Encoder Layer
        self.enc_layers.append(
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         activation='relu',
                         dropout_rate=dropout_rate,
                         return_attention_scores=True))
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # x will be in (batch_size, n_patches, sequense length)
        x = self.pos_projection(x)
        # x Shape `(batch_size, N+d_dp_rep, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)  # shape (B, N+1, d_model)

        self.last_attn_scores = self.enc_layers[-1].last_attn_scores

        return x  # Shape `(batch_size, N+1, d_model)`.

    def get_config(self):
        config = super(DeepPhenotypeLobeEncoder, self).get_config()
        config['d_model'] = self.d_model
        config['d_dp_rep'] = self.d_dp_rep
        config['num_layers'] = self.num_layers
        return config


def create_LobTe(input_shape,
                 num_layers,
                 d_model,
                 d_dp_rep,
                 num_heads,
                 dff,
                 d_deepfeatures,
                 oname,
                 name='LobeTransformer',
                 patch_size=(300, 11),
                 dropout_rate=0.25):

    def input_to_patches(x, patch_size):
        # We are going to use patches of (60,11) for the deep_phenotype for
        # each lobe, so we are going to have Nlobes * (300//60) * (11//11)
        # patches to and the the flattened patch will be of 660 pixels
        B, L, H, W, C = x.shape
        np1 = H // patch_size[0]
        np2 = W // patch_size[1]
        lnp = L * np1 * np2
        x = tf.reshape(x, (-1, L, np1, patch_size[0], np2, patch_size[1], C))
        # Batch shape is None so we should use -1
        # Transpose (B,L,H_patches,Hp_size, W_patches,Wp_size,C) to
        # (B,L,H_patches, W_patches, Hp_size,Wp_size,C)
        x = tf.transpose(x, (0, 1, 2, 4, 3, 5, 6))
        # Reshape to (B,L*H_patches*W_patches, Hp_size,Wp_size,C)
        x = tf.reshape(x, (-1, lnp, patch_size[0], patch_size[1], C))
        # Reshape to (B, Npatches, Hp_size*Wp_zie*C)
        x = tf.reshape(x, (-1, lnp, patch_size[0] * patch_size[1] * C))
        return x

    ti = tf.keras.layers.Input(shape=input_shape)
    t1 = input_to_patches(ti, patch_size)
    # The deep_phenotype representation learned is in the position 0:d_dp_drep
    encoder = DeepPhenotypeLobeEncoder(num_layers,
                                       d_model,
                                       d_dp_rep,
                                       num_heads,
                                       dff,
                                       dropout_rate=dropout_rate)(t1)
    # Grab the learned deep_phenotype representation from the encoder.
    dp_rep = encoder[:, :d_dp_rep]  # (batch_size, d_dp_rep, d_model)
    flatten = tf.keras.layers.Flatten()(dp_rep)
    deep_features = tf.keras.layers.Dense(d_deepfeatures)(flatten)
    deep_features = tf.keras.layers.Activation('gelu')(deep_features)
    deep_features = tf.keras.layers.BatchNormalization(
        name='deep_features')(deep_features)

    output = tf.keras.layers.Dense(1, name=oname)(deep_features)
    model = tf.keras.models.Model(ti, output, name=name)
    return model
