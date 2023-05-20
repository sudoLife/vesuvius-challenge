import tensorflow as tf
import numpy as np

import random
import gc
import cv2
import time
from tqdm import tqdm
from skimage import exposure


class Pipeline:

    def __init__(self, data_dir, patch_size, downsampling, z_dim, z_start, batch_size):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.patch_halfsize = patch_size // 2
        self.downsampling = downsampling
        self.z_dim = z_dim
        self.z_start = z_start
        self.batch_size = batch_size

    def get_input_shape(self):
        return (self.patch_size, self.patch_size, self.z_dim)

    def resize(self, img):
        if self.downsampling != 1.:
            size = int(img.shape[1] * self.downsampling), int(img.shape[0] * self.downsampling)
            img = cv2.resize(img, size)
        return img

    def load_mask(self, split, index):
        img = cv2.imread(f"{self.data_dir}/{split}/{index}/mask.png", 0)
        img = self.resize(img)
        return img.astype("bool")

    def load_labels(self, split, index):
        img = cv2.imread(f"{self.data_dir}/{split}/{index}/inklabels.png", 0)
        img = self.resize(img)
        return np.expand_dims(img, axis=-1)

    def load_surface_volume(self, split, index):
        """Loads surface volume

        Args:
            split (str): train or test
            index (str|int): folder index

        Returns:
            np.ndarray: volumes
        """
        fnames = [f"{self.data_dir}/{split}/{index}/surface_volume/{i:02}.tif"
                  for i in range(self.z_start, self.z_start + self.z_dim)]

        # NOTE: this batch size doesn't really have anything to do with the training batch size
        fname_batches = [fnames[i:i + self.batch_size] for i in range(0, len(fnames), self.batch_size)]
        # NOTE: wouldn't it be faster to pre-allocate this with a numpy array?
        volumes = []
        for fname_batch in fname_batches:
            z_slices = []
            for fname in tqdm(fname_batch):
                # shape of (height, width) with values between 0 and 1
                img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
                img = self.resize(img)
                img = (exposure.equalize_adapthist(img) * 255.0).astype('uint8')
                z_slices.append(img)
            volumes.append(np.stack(z_slices, axis=-1))
            del z_slices
            gc.collect()
        return np.concatenate(volumes, axis=-1)

    def load_sample(self, split, index):
        """Load data from a specified index folder

        Args:
            split (str): train or test. For test, won't load the labels
            index (int|str): folder index

        Returns:
            tuple: volume, mask, labels if train, else volume, mask
        """
        print(f"Loading '{split}/{index}'...")
        gc.collect()
        if split == "train":
            return self.load_surface_volume(
                split, index), self.load_mask(
                split, index), self.load_labels(
                split, index)
        return self.load_surface_volume(split, index), self.load_mask(split, index), None

    def sample_random_location(self, shape):
        """Sample a random location on the image

        Args:
            shape (iterable): shape of the image

        Returns:
            tuple: x and y coordinates
        """
        x = random.randint(self.patch_halfsize, shape[0] - self.patch_halfsize - 1)
        y = random.randint(self.patch_halfsize, shape[1] - self.patch_halfsize - 1)
        return (x, y)

    def list_all_locations(self, mask, stride=None):
        if stride is None:
            stride = self.patch_halfsize
        locations = []
        for x in range(self.patch_halfsize, mask.shape[0] - self.patch_halfsize, stride):
            for y in range(self.patch_halfsize, mask.shape[1] - self.patch_halfsize, stride):
                if mask[x, y]:
                    locations.append((x, y))
        return locations

    def extract_patch(self, location, volume):
        x = location[0]
        y = location[1]
        patch = volume[x - self.patch_halfsize:x + self.patch_halfsize,
                       y - self.patch_halfsize:y + self.patch_halfsize, :]
        return patch.astype("float32") / 255.0

    def extract_labels(self, location, labels):
        x = location[0]
        y = location[1]

        label = labels[x - self.patch_halfsize:x + self.patch_halfsize,
                       y - self.patch_halfsize:y + self.patch_halfsize, :]
        return label.astype("float32") / 255.0

    def make_random_data_generator(self, volume, mask, labels):
        """Create an endless sampling of patches from random locations

        Args:
            volume (np.ndarray): input volume
            mask (np.ndarray): volume mask
            labels (np.ndarray): labels
        """
        def data_generator():
            while True:
                loc = self.sample_random_location(mask.shape)
                # check if the center is in the mask
                if mask[loc[0], loc[1]]:
                    patch = self.extract_patch(loc, volume)
                    label = self.extract_labels(loc, labels)
                    yield patch, label
        return data_generator

    def make_iterated_data_generator(self, volume, mask, labels=None):
        locations = self.list_all_locations(mask)
        print(f"Iterated dataset size: {len(locations)}")

        def data_generator():
            for loc in locations:
                patch = self.extract_patch(loc, volume)
                if labels is None:
                    yield patch
                else:
                    label = self.extract_labels(loc, labels)
                    yield patch, label
        return data_generator

    def make_tf_dataset(self, gen_fn, labeled=True):
        if labeled:
            output_signature = (
                tf.TensorSpec(shape=(self.patch_size, self.patch_size, self.z_dim), dtype=tf.float32),
                tf.TensorSpec(shape=(self.patch_size, self.patch_size, 1), dtype=tf.float32),
            )
        else:
            output_signature = tf.TensorSpec(shape=(self.patch_size, self.patch_size, self.z_dim), dtype=tf.float32)
        ds = tf.data.Dataset.from_generator(
            gen_fn,
            output_signature=output_signature,
        )
        return ds.prefetch(tf.data.AUTOTUNE).batch(self.batch_size)

    def make_datasets_for_fold(self, fold, train_augment_fn=None):
        train_volumes = fold["train_volumes"]
        train_masks = fold["train_masks"]
        train_labels = fold["train_labels"]

        include_validation = "validation_volume" in fold
        if include_validation:
            validation_volume = fold["validation_volume"]
            validation_mask = fold["validation_mask"]
            validation_labels = fold["validation_labels"]

        all_train_ds = []
        for volume, mask, labels in zip(train_volumes, train_masks, train_labels):
            train_ds = self.make_tf_dataset(
                self.make_random_data_generator(volume, mask, labels),
                labeled=True,
            )
            all_train_ds.append(train_ds)
        train_ds = tf.data.Dataset.sample_from_datasets(all_train_ds)

        if train_augment_fn:
            train_ds = train_ds.map(train_augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        if not include_validation:
            return train_ds

        val_ds = self.make_tf_dataset(
            self.make_iterated_data_generator(validation_volume, validation_mask, validation_labels),
            labeled=True,
        )
        return train_ds, val_ds

    def check_throughout(self, ds):
        n = 100
        for i, _ in enumerate(ds.take(n + 1)):
            if i == 1:  # Don't include dataset initialization time
                t0 = time.time()
        time_per_batch = (time.time() - t0) / n
        print(f"Time per batch: {time_per_batch:.4f}s")
        print(f"Time per sample: {time_per_batch / self.batch_size:.4f}s")
