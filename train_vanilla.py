from train_utils import *
from pipeline import *
import segmentation_models as sm
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import os


def main():

    train_transform = augment_albumentations()

    data_params = {
        "data_dir": "./dataset/",
        "patch_size": 512,
        "downsampling": 1.0,
        # "layers": 40,
        # "z_start": 0,
        'layers': list(range(11, 24)) + list(range(27, 36)),
        "batch_size": 4,
        'train_transform': train_transform,  # either None or this
        'use_adapt_hist': False,
        'subtract_ind': None
    }

    epochs = 200
    steps_per_epoch = 100
    backbone = 'no_backbone'  # 'resnet18'

    # NOTE: set this to something memorable as it's going to name the log folder
    title = backbone + "_vanilla_unet_5_levels_10_filter_size"

    pipeline = Pipeline(**data_params)

    volume_1, mask_1, labels_1 = pipeline.load_sample(split="train", index=1)
    volume_2, mask_2, labels_2 = pipeline.load_sample(split="train", index=2)
    volume_3, mask_3, labels_3 = pipeline.load_sample(split="train", index=3)

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

    dev = 'dev_2'
    # for now let's only
    train_ds, val_ds = pipeline.make_datasets_for_fold(dev_folds[dev])

    val_step = len(pipeline.list_all_locations(dev_folds[dev]['validation_mask'])) // data_params['batch_size']

    # model = create_basic_unet_model(pipeline.get_input_shape(), [64, 128, 256, 512, 1024], filter_size=(3, 3))
    model = create_basic_unet_model(pipeline.get_input_shape(), [16, 32, 64, 128, 256], filter_size=(3, 3))

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)

    model.compile(optimizer=optimizer, loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score])

    logdir_current = 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S") + title

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir_current + '/scalars')
    callback_chkpt = tf.keras.callbacks.ModelCheckpoint(
        logdir_current + '/model.h5',
        monitor='val_iou_score',
        mode='max',
        save_weights_only=False,
        # save_best_only=True,
        save_freq=10 * steps_per_epoch,
    )

    if not os.path.exists(logdir_current):
        # Create the folder
        os.makedirs(logdir_current)

    tf.keras.utils.plot_model(model, logdir_current + '/model.png')

    model.fit(
        train_ds, batch_size=data_params['batch_size'],
        epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=val_step,
        callbacks=[callback_chkpt, tensorboard_callback])

    model.save_weights(logdir_current + '/final_checkpoint')
    print("Done")


if __name__ == "__main__":
    main()
