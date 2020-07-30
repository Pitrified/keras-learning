import argparse
import os
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from random import seed as rseed
from timeit import default_timer as timer


def parse_arguments():
    """Setup CLI interface
    """
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i",
        "--path_input",
        type=str,
        default="hp.jpg",
        help="path to input image to use",
    )

    parser.add_argument(
        "-s", "--rand_seed", type=int, default=-1, help="random seed to use"
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_logger(logLevel="DEBUG"):
    """Setup logger that outputs to console for the module
    """
    logroot = logging.getLogger("c")
    logroot.propagate = False
    logroot.setLevel(logLevel)

    module_console_handler = logging.StreamHandler()

    #  log_format_module = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    #  log_format_module = "%(name)s - %(levelname)s: %(message)s"
    #  log_format_module = '%(levelname)s: %(message)s'
    #  log_format_module = '%(name)s: %(message)s'
    log_format_module = "%(message)s"

    formatter = logging.Formatter(log_format_module)
    module_console_handler.setFormatter(formatter)

    logroot.addHandler(module_console_handler)

    logging.addLevelName(5, "TRACE")
    # use it like this
    # logroot.log(5, 'Exceedingly verbose debug')


def setup_env():
    setup_logger()

    args = parse_arguments()

    # setup seed value
    if args.rand_seed == -1:
        myseed = 1
        myseed = int(timer() * 1e9 % 2 ** 32)
    else:
        myseed = args.rand_seed
    rseed(myseed)
    np.random.seed(myseed)

    # build command string to repeat this run
    # FIXME if an option is a flag this does not work, sorry
    recap = f"python3 ex_load_dataset_mnist.py"
    for a, v in args._get_kwargs():
        if a == "rand_seed":
            recap += f" --rand_seed {myseed}"
        else:
            recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def load_image(file_path):
    """TODO: what is load_image doing?
    """
    # logg = logging.getLogger(f"c.{__name__}.load_image")
    # logg.debug(f"Start load_image")

    label = tf.strings.split(file_path, os.sep)[-2]
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image = tf.reshape(image, (28, 28))
    image = tf.reshape(image, (784,))
    # image = tf.image.resize(image, [28, 28])
    return image, int(label)


def load_dataset(ds_root_folder):
    """TODO: what is load_dataset doing?
    """
    logg = logging.getLogger(f"c.{__name__}.load_dataset")
    logg.debug(f"Start load_dataset")

    list_ds = tf.data.Dataset.list_files(str(ds_root_folder / "*/*"))

    for f in list_ds.take(5):
        print(f.numpy())

    labeled_ds = list_ds.map(load_image)
    for image_raw, label_text in labeled_ds.take(1):
        print(image_raw)
        print(label_text.numpy())

    batch_size = 1024
    batched_ds = labeled_ds.batch(batch_size)

    prefetch_ds = batched_ds.prefetch(3)

    return prefetch_ds


def run_ex_load_dataset_mnist(args):
    """TODO: What is ex_load_dataset_mnist doing?

    https://www.tensorflow.org/guide/data
    https://www.tensorflow.org/api_docs/python/tf/data/Dataset#list_files
    """
    logg = logging.getLogger(f"c.{__name__}.run_ex_load_dataset_mnist")
    logg.debug(f"Starting run_ex_load_dataset_mnist")

    mnist_root_train = Path("../datasets/mnist_png/training")
    mnist_root_test = Path("../datasets/mnist_png/testing")
    train_ds = load_dataset(mnist_root_train)
    test_ds = load_dataset(mnist_root_test)

    inputs = keras.Input(shape=(784,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    model.summary()

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    model.fit(train_ds, epochs=2)

    test_scores = model.evaluate(test_ds, verbose=2)
    logg.debug(f"Test loss: {test_scores[0]}")
    logg.debug(f"Test accuracy: {test_scores[1]}")


if __name__ == "__main__":
    args = setup_env()
    run_ex_load_dataset_mnist(args)
