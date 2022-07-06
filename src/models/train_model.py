import logging
from pathlib import Path
import click
import tensorflow as tf
import numpy as np
import yaml
from dvclive.keras import DvcLiveCallback


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


params = yaml.safe_load(open("params.yaml"))["train"]


def reset_random_seeds(seed):
    import os

    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    import random

    random.seed(seed)


@click.command()
@click.argument("data_filepath", type=click.Path(exists=True))
@click.argument("model_filepath", type=click.Path())
@click.argument("report_filepath", type=click.Path())
def main(data_filepath, model_filepath, report_filepath):
    """Runs data processing scripts to turn raw data from (data/external) into
    cleaned data ready to be trained (saved in data/processed).
    """

    reset_random_seeds(params["seed"])

    def load_data(filepath):
        data = np.load(filepath)
        ds = tf.data.Dataset.from_tensor_slices((data["images"], data["labels"]))
        ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(params["batch_size"])
        ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    data_filepath = Path(data_filepath)
    ds_train = load_data(data_filepath / "train.npz")
    ds_val = load_data(data_filepath / "val.npz")
    ds_test = load_data(data_filepath / "test.npz")

    model = tf.keras.models.Sequential(
        [
            # tf.keras.layers.Flatten(input_shape=(28, 28)),
            # tf.keras.layers.Dense(128, activation="relu"),
            # tf.keras.layers.Dense(10),
            tf.keras.layers.Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=params["epochs"],
        validation_data=ds_val,
        callbacks=[DvcLiveCallback(path=report_filepath)],
    )

    model.save(model_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
