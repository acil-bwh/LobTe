"""
File: schedules.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
"""

import tensorflow as tf


class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        d_model = tf.cast(self.d_model, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {'d_model': self.d_model, 'warmup_steps': self.warmup_steps}
        return config
