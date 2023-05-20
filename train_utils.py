import tensorflow as tf
import numpy as np
from tqdm import tqdm


@tf.function
def train_augment_fn(patch, labels):
    patch = tf.image.random_flip_left_right(patch, seed=1337)
    labels = tf.image.random_flip_left_right(labels, seed=1337)

    patch = tf.image.random_flip_up_down(patch, seed=42)
    labels = tf.image.random_flip_up_down(labels, seed=42)
    return patch, labels


def predict_and_assemble(pipeline, volume, mask, threshold, model):
    test_a_ds = pipeline.make_iterated_data_generator(volume, mask)
    locations = pipeline.list_all_locations(mask)
    all_pred = np.zeros(mask.shape)
    all_binary_pred = np.zeros(mask.shape)
    pred = []

    for i, patch in tqdm(enumerate(test_a_ds())):
        p = model.predict(np.expand_dims(patch, 0))[0]
        pred.append(p)
        c = locations[i]
        l = c[0] - pipeline.patch_halfsize
        r = c[0] + pipeline.patch_halfsize
        t = c[1] + pipeline.patch_halfsize
        b = c[1] - pipeline.patch_halfsize
        p = p.squeeze()
        all_pred[l:r, b:t] = p
        p = p > threshold
        all_binary_pred[l:r, b:t] = p

    return all_pred, all_binary_pred, pred
