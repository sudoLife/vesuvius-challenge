import tensorflow as tf


@tf.function
def train_augment_fn(patch, labels):
    patch = tf.image.random_flip_left_right(patch, seed=1337)
    labels = tf.image.random_flip_left_right(labels, seed=1337)

    patch = tf.image.random_flip_up_down(patch, seed=42)
    labels = tf.image.random_flip_up_down(labels, seed=42)
    return patch, labels
