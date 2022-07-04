import logging
from pathlib import Path
import click
import tensorflow as tf
import numpy as np
from dvclive import Live
from sklearn import metrics


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


def evaluate(live, model, data, key):

    loss, acc = model.evaluate(data["images"], data["labels"], verbose=2)
    live.log(f"{key}/loss", float(loss))
    live.log(f"{key}/accuracy", float(acc))
    # test_predictions = model.predict(data["images"])

    # live.log_plot("roc", data["labels"], test_predictions)
    # live.log("avg_prec", metrics.average_precision_score(data["labels"], test_predictions))
    # live.log("roc_auc", metrics.roc_auc_score(data["labels"], test_predictions))
    # live.log("confusion_matrix", metrics.roc_auc_score(data["labels"], test_predictions))


@click.command()
@click.argument("data_filepath", type=click.Path(exists=True))
@click.argument("model_filepath", type=click.Path(exists=True))
@click.argument("result_filepath", type=click.Path())
def main(data_filepath, model_filepath, result_filepath):
    """Runs data processing scripts to turn raw data from (data/external) into
    cleaned data ready to be trained (saved in data/processed).
    """
    live = Live(result_filepath)

    model = tf.keras.models.load_model(model_filepath)

    def load_data(filepath):
        data = np.load(filepath)
        return data
        # ds = tf.data.Dataset.from_tensor_slices((data["images"], data["labels"]))
        # ds = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        # ds = ds.batch(batch_size)
        # ds = ds.cache()
        # ds = ds.prefetch(tf.data.AUTOTUNE)
        # return ds

    data_filepath = Path(data_filepath)
    ds_train = load_data(data_filepath / "train.npz")
    ds_val = load_data(data_filepath / "val.npz")
    ds_test = load_data(data_filepath / "test.npz")

    # evaluate(model, ds_train, result_filepath / "train")
    result_filepath = Path(result_filepath)
    result_filepath.mkdir(exist_ok=True)
    evaluate(live, model, ds_train, "train")
    evaluate(live, model, ds_val, "val")
    evaluate(live, model, ds_test, "test")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
