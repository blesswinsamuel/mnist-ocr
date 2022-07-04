# -*- coding: utf-8 -*-
import sys
import click
import logging
from pathlib import Path

# from dotenv import find_dotenv, load_dotenv

import tensorflow as tf
import numpy as np
import idx2numpy
import gzip
from sklearn.model_selection import train_test_split
import yaml

# Reference: https://www.tensorflow.org/datasets/keras_example

params = yaml.safe_load(open("params.yaml"))["prepare_data"]


def load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        return idx2numpy.convert_from_file(f)


def load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        return idx2numpy.convert_from_file(f)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (data/external) into
    cleaned data ready to be trained (saved in data/processed).
    """
    input_filepath = Path(input_filepath)

    train_images = load_images(input_filepath / "train-images-idx3-ubyte.gz")
    train_labels = load_labels(input_filepath / "train-labels-idx1-ubyte.gz")
    test_images = load_images(input_filepath / "t10k-images-idx3-ubyte.gz")
    test_labels = load_labels(input_filepath / "t10k-labels-idx1-ubyte.gz")

    print(f"Found {len(train_images)} train images")
    print(f"Found {len(train_labels)} train labels")
    print(f"Found {len(test_images)} test images")
    print(f"Found {len(test_labels)} test labels")

    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=params["val_split"], random_state=params["seed"]
    )

    output_filepath = Path(output_filepath)
    output_filepath.mkdir(exist_ok=True)
    np.savez_compressed(output_filepath / "train.npz", images=train_images, labels=train_labels)
    np.savez_compressed(output_filepath / "val.npz", images=val_images, labels=val_labels)
    np.savez_compressed(output_filepath / "test.npz", images=test_images, labels=test_labels)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # # find .env automagically by walking up directories until it's found, then
    # # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
