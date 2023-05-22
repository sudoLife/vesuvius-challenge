import tensorflow as tf
import numpy as np
from tqdm import tqdm
import albumentations as A
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model


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


def augment_albumentations():
    transforms = A.Compose([
        A.Rotate(),
        A.VerticalFlip(),
        A.GridDistortion()
    ])

    return transforms


def create_basic_unet_model(input_shape, filter_nums, filter_size=(3, 3), conv_act='relu', final_act='sigmoid'):
    # variable number of input channels
    inputs = Input(input_shape)

    def encoding_block(inputs, filter_num):
        conv = Conv2D(filter_num, filter_size, activation=conv_act, padding='same')(inputs)
        conv = Conv2D(filter_num, filter_size, activation=conv_act, padding='same')(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)

        return conv, pool

    def decoding_block(inputs, skip_connection, filter_num):
        up = concatenate([UpSampling2D(size=(2, 2))(inputs), skip_connection], axis=3)
        conv = Conv2D(filter_num, filter_size, activation=conv_act, padding='same')(up)
        conv = Conv2D(filter_num, filter_size, activation=conv_act, padding='same')(conv)

        return conv

    skip_conns = []
    conv = inputs

    for filter_num in filter_nums[:-1]:
        conv, pool = encoding_block(conv, filter_num)
        skip_conns.append(conv)
        conv = pool

    # bottom
    conv = Conv2D(filter_nums[-1], filter_size, activation=conv_act, padding='same')(conv)

    for i in reversed(range(len(skip_conns))):
        conv = decoding_block(conv, skip_conns[i], filter_nums[i])

    # output
    output = Conv2D(1,  (1, 1), activation=final_act)(conv)

    model = Model(inputs=[inputs], outputs=output)
    model.summary()
    return model
