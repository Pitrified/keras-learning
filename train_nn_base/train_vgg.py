import argparse
import logging
import cv2
import matplotlib.pyplot as plt
import pickle
import json

import numpy as np

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from os import listdir
from os.path import abspath
from os.path import dirname
from os.path import join
from random import seed
from timeit import default_timer as timer

from smallVGGnet import SmallVGGNet
from middleVGGnet import MiddleVGGNet


def parse_arguments():
    """Setup CLI interface
    """
    parser = argparse.ArgumentParser(description="Train a VGG-like net on a dataset")

    parser.add_argument(
        "-pi",
        "--path_input",
        type=str,
        #  default="~/coin-dataset/data_mod_64_split/train",
        default="../../coin-dataset/data_mod_64_split/train",
        help="path to input images to use",
    )

    ap.add_argument(
        "-w", "--width", type=int, default=64, help="target spatial dimension width"
    )
    ap.add_argument(
        "-e", "--height", type=int, default=64, help="target spatial dimension height"
    )

    parser.add_argument(
        "-b",
        "--basename",
        required=True,
        type=str,
        help="basename for model, labels and plot: {path/to/basename}.model",
    )

    parser.add_argument(
        "-v",
        "--vgg_size",
        required=True,
        choices=["small", "middle"],
        type=str,
        help="which VGG to train, either small or middle",
    )

    parser.add_argument(
        "-fc",
        "--fc_size",
        type=int,
        default=-1,
        help="size of the fully connected layers",
    )

    # last line to parse the args
    args = parser.parse_args()
    return args


def setup_logger(logLevel="DEBUG"):
    """Setup logger that outputs to console for the module
    """
    logmoduleconsole = logging.getLogger(f"{__name__}.console")
    logmoduleconsole.propagate = False
    logmoduleconsole.setLevel(logLevel)

    module_console_handler = logging.StreamHandler()

    #  log_format_module = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_format_module = "%(asctime)s - %(levelname)s - %(message)s"
    #  log_format_module = "%(name)s - %(levelname)s: %(message)s"
    #  log_format_module = '%(levelname)s: %(message)s'
    #  log_format_module = "%(message)s"

    formatter = logging.Formatter(log_format_module)
    module_console_handler.setFormatter(formatter)

    logmoduleconsole.addHandler(module_console_handler)

    logging.addLevelName(5, "TRACE")
    # use it like this
    # logmoduleconsole.log(5, 'Exceedingly verbose debug')

    return logmoduleconsole


def load_dataset(path_dataset, width, height, logLevel="WARN"):
    """Load the dataset found in path_dataset, each subfolder is a label
    """
    logload = logging.getLogger(f"{__name__}.console.trainvgg")
    logload.setLevel(logLevel)

    tot_images = 0
    for label in listdir(path_dataset):
        label_full = join(path_dataset, label)
        for img_name in listdir(label_full):
            tot_images += 1

    logload.info(f"Loading {tot_images} images")

    # allocate the memory
    # THE DTYPE is float, should be the right one
    all_images = np.zeros((tot_images, width, height, 3))

    true_labels = []
    num_image = 0
    for label in listdir(path_dataset):
        label_full = join(path_dataset, label)
        for img_name in listdir(label_full):
            #  for img_name in listdir(label_full)[:10]:
            img_name_full = join(label_full, img_name)
            logload.log(5, f"Opening {img_name_full} {width}")

            image = cv2.imread(img_name_full)

            image = cv2.resize(image, (width, height))

            # scale the pixel values to [0, 1]
            #  image = image.astype("float") / 255.0

            all_images[num_image, :, :, :] = image

            num_image += 1
            true_labels.append(label)

    all_images = all_images.astype("float") / 255.0
    logload.debug(f"All_images.shape {all_images.shape}")

    #  cv2.imshow('Resized all_images[0]', all_images[0])
    #  cv2.waitKey(0)

    # XXX true_labels might need to be np.array(true_labels)
    return all_images, true_labels


def do_vgg_train(
    path_input, width, height, basename, vgg_size, fc_size, logLevel="WARN"
):
    """Train a VGG-like convolutional network
    """
    logvgg = logging.getLogger(f"{__name__}.console.trainvgg")
    logvgg.setLevel(logLevel)

    model_file = f"{basename}.model"
    label_bin_file = f"{basename}.pickle"
    plot_file = f"{basename}.png"
    logvgg.debug(f"mf {model_file} lbf {label_bin_file} pf {plot_file}")

    data, labels = load_dataset(path_input, width, height, "INFO")

    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25)

    # convert the labels from integers to vectors (for 2-class, binary
    # classification you should use Keras' to_categorical function
    # instead as the scikit-learn's LabelBinarizer will not return a
    # vector)
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    # construct the image generator for data augmentation
    # rotation is ok, shear/shift/flip reduced
    aug = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.01,
        height_shift_range=0.01,
        shear_range=0.002,
        zoom_range=0.02,
        horizontal_flip=False,
        fill_mode="nearest",
    )

    if vgg_size == "small":
        # TODO fc_size set from here
        model = SmallVGGNet.build(
            width=width, height=height, depth=3, classes=len(lb.classes_)
        )
    elif vgg_size == "middle":
        # default value of fc_size
        if fc_size == -1:
            fc_size = 512
        model = MiddleVGGNet.build(
            width=width,
            height=height,
            depth=3,
            classes=len(lb.classes_),
            fully_connected_size=fc_size,
        )
    else:
        logvgg.critical(f"Unrecognized dimension {vgg_size}, stopping.")
        return -1

    # initialize our initial learning rate, # of epochs to train for, and batch size
    INIT_LR = 0.01
    EPOCHS = 75
    #  EPOCHS = 3
    BS = 32
    # TODO fiddle with this

    # initialize the model and optimizer (you'll want to use
    # binary_crossentropy for 2-class classification)
    logvgg.info("Training network...")
    opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    # TODO fiddle with this

    # save model summary
    summary_file = f"{basename}_summary.txt"
    with open(summary_file, "w") as sf:
        model.summary(line_length=100, print_fn=lambda x: sf.write(f"{x}\n"))
        # using an actual logger: print_fn=logger.info

    # save the model structure in JSON format
    config = model.get_config()
    config_json_file = f"{basename}_structure.json"
    with open(config_json_file, "w") as jf:
        json.dump(config, jf)

    # train the network
    H = model.fit_generator(
        aug.flow(trainX, trainY, batch_size=BS),
        validation_data=(testX, testY),
        steps_per_epoch=len(trainX) // BS,
        epochs=EPOCHS,
    )

    # save the model and label binarizer to disk
    logvgg.info("Serializing network and label binarizer...")
    model.save(model_file)
    with open(label_bin_file, "wb") as f:
        f.write(pickle.dumps(lb))

    # evaluate the network
    logvgg.info("Evaluating network...")
    predictions = model.predict(testX, batch_size=32)
    report = classification_report(
        testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_
    )
    logvgg.info(f"\n{report}")
    report_file = f"{basename}_report.txt"
    with open(report_file, "w") as rf:
        rf.write(report)

    # plot the training loss and accuracy
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["acc"], label="train_acc")
    plt.plot(N, H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy (SmallVGGNet)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(plot_file)


def main():
    logmoduleconsole = setup_logger()

    args = parse_arguments()

    path_input = abspath(args.path_input)
    width = args.width
    height = args.height
    basename = args.basename
    vgg_size = args.vgg_size
    fc_size = args.fc_size

    recap = f"python3 train_vgg.py"
    recap += f" --path_input {path_input}"
    recap += f" --width {width}"
    recap += f" --height {height}"
    recap += f" --basename {basename}"
    recap += f" --vgg_size {vgg_size}"
    recap += f" --fc_size {fc_size}"

    logmoduleconsole.info(recap)

    do_vgg_train(path_input, width, height, basename, vgg_size, fc_size)


if __name__ == "__main__":
    main()
