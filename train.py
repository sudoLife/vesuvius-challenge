from train_utils import *
from pipeline import *
import segmentation_models as sm
import matplotlib.pyplot as plt
import pickle
from datetime import datetime


def main():
    data_dir = "./dataset/"
    patch_size = 800  # e.g. 128x128
    downsampling = 1.0  # setting this to e.g. 0.5 means images will be loaded as 2x smaller. 1 does nothing.
    z_dim = 40   # number of slices in the z direction. max value is 65 - z_start
    z_start = 0  # offset of slices in the z direction
    batch_size = 4
    epochs = 200
    steps_per_epoch = 100
    # this means it will take val_step batches
    backbone = 'resnet18'  # 'resnet18'

    train_transform = augment_albumentations()

    pipeline = Pipeline(data_dir, patch_size, downsampling, z_dim, z_start, batch_size,
                        train_transform=train_transform, use_adapt_hist=False)

    preprocessing = sm.get_preprocessing(backbone)
    # def preprocessing(x): return x

    volume_1, mask_1, labels_1 = pipeline.load_sample(split="train", index=1)
    volume_1 = preprocessing(volume_1)
    volume_2, mask_2, labels_2 = pipeline.load_sample(split="train", index=2)
    volume_2 = preprocessing(volume_2)
    volume_3, mask_3, labels_3 = pipeline.load_sample(split="train", index=3)
    volume_3 = preprocessing(volume_3)

    gc.collect()
    print("Loading complete.")

    # NOTE: maybe get rid of
    dev_folds = {
        "dev_1": {
            "train_volumes": [volume_1, volume_2],
            "train_labels": [labels_1, labels_2],
            "train_masks": [mask_1, mask_2],
            "validation_volume": volume_3,
            "validation_labels": labels_3,
            "validation_mask": mask_3,
        },
        "dev_2": {
            "train_volumes": [volume_1, volume_3],
            "train_labels": [labels_1, labels_3],
            "train_masks": [mask_1, mask_3],
            "validation_volume": volume_2,
            "validation_labels": labels_2,
            "validation_mask": mask_2,
        },
        "dev_3": {
            "train_volumes": [volume_2, volume_3],
            "train_labels": [labels_2, labels_3],
            "train_masks": [mask_2, mask_3],
            "validation_volume": volume_1,
            "validation_labels": labels_1,
            "validation_mask": mask_1,
        }
    }

    gc.collect()

    # for now let's only
    train_ds, val_ds = pipeline.make_datasets_for_fold(dev_folds['dev_1'])

    val_step = len(pipeline.list_all_locations(dev_folds['dev_1']['validation_mask'])) // batch_size

    # If you need to specify non-standard input shape
    model = sm.Unet(
        backbone,
        input_shape=pipeline.get_input_shape(),
        encoder_weights=None,
        classes=1
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007)

    model.compile(optimizer=optimizer, loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

    title = backbone + "no_adapthist_16_volumes_from_middle"

    logdir_current = 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S") + title

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir_current + '/scalars')
    callback_chkpt = tf.keras.callbacks.ModelCheckpoint(
        logdir_current + '/checkpoint',
        monitor='loss',
        mode='min',
        save_weights_only=True,
        save_freq=steps_per_epoch * 10,
    )

    history = model.fit(train_ds, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch,
                        validation_data=val_ds, validation_steps=val_step, callbacks=[callback_chkpt, tensorboard_callback])

    # fmtstr = f'unet2d_b{batch_size}_p{patch_size}_d{downsampling}_zdim{z_dim}_e{epochs}_spe{steps_per_epoch}'
    model.save_weights(logdir_current + '/final_checkpoint')
    print("Done")


if __name__ == "__main__":
    main()
