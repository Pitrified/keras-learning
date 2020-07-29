import argparse
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
    recap = f"python3 keras_functional.py"
    for a, v in args._get_kwargs():
        if a == "rand_seed":
            recap += f" --rand_seed {myseed}"
        else:
            recap += f" --{a} {v}"

    logmain = logging.getLogger(f"c.{__name__}.setup_env")
    logmain.info(recap)

    return args


def run_keras_functional(args):
    """TODO: What is keras_functional doing?

    Some examples with the Functional API
    https://www.tensorflow.org/guide/keras/functional/
    """
    logg = logging.getLogger(f"c.{__name__}.run_keras_functional")
    logg.debug(f"Starting run_keras_functional")

    img_inputs = keras.Input(shape=(32, 32, 3))
    inputs = keras.Input(shape=(784,))
    logg.debug(f"inputs.shape: {inputs.shape}")
    logg.debug(f"inputs.dtype: {inputs.shape}")

    # create a new node in the graph of layers by calling a layer on this inputs object:

    dense = layers.Dense(64, activation="relu")
    x = dense(inputs)

    # "layer call" action is like drawing an arrow from "inputs" to this layer you created. You're "passing" the inputs to the dense layer, and out you get x

    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(10)(x)

    # create a Model by specifying its inputs and outputs in the graph of layers:

    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")

    model.summary()

    keras.utils.plot_model(model, "my_first_model.png")
    keras.utils.plot_model(model, "my_first_model_shape_info.png", show_shapes=True)

    # download it (will be cached in ~/.keras/datasets
    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # local dataset
    mnist_path = Path("../datasets/mnist.npz")
    logg.debug(f"mnist_path: {mnist_path}")
    with np.load(mnist_path) as data:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]
        logg.debug(f"x_train.shape: {x_train.shape}")
        logg.debug(f"y_train[0]: {y_train[0]}")

    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )

    history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    logg.debug(f"Test loss: {test_scores[0]}")
    logg.debug(f"Test accuracy: {test_scores[1]}")


if __name__ == "__main__":
    args = setup_env()
    run_keras_functional(args)
