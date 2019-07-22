from keras.models import load_model
from os import listdir
from os.path import join

import numpy as np

import argparse
import pickle
import cv2


def load_dataset(path_test, width, height):
    """Load the dataset found in path_test, each folder is a label
    """
    tot_images = 0
    for label in listdir(path_test):
        label_full = join(path_test, label)
        for img_name in listdir(label_full):
            tot_images += 1

    # allocate the memory
    # THE DTYPE is float, should be the right one
    all_images = np.zeros((tot_images, width, height, 3))

    true_labels = []
    num_images = 0
    for label in listdir(path_test):
        label_full = join(path_test, label)
        for img_name in listdir(label_full):
            #  for img_name in listdir(label_full)[:10]:
            img_name_full = join(label_full, img_name)
            print(f"Opening {img_name_full} {width}")

            image = cv2.imread(img_name_full)

            image = cv2.resize(image, (width, height))

            # scale the pixel values to [0, 1]
            image = image.astype("float") / 255.0

            all_images[num_images, :, :, :] = image

            num_images += 1
            true_labels.append(label)

    print(f"All_images.shape {all_images.shape}")

    #  cv2.imshow('Resized all_images[0]', all_images[0])
    #  cv2.waitKey(0)

    return all_images, true_labels


def load_model_label(model_path, label_bin_path):
    """Load the model and the pickled labels
    """
    # load the model and label binarizer
    print("[INFO] loading network and label binarizer...")
    model = load_model(model_path)
    lb = pickle.loads(open(label_bin_path, "rb").read())
    return model, lb


def compute_confusion_matrix(model, lb, all_images, true_labels):
    """Evaluate model performance on test images

    Precision: TP / ( TP + FP)
    Recall: TP / ( TP + FN)
    F-score: 2 (PxR) / (P+R)
    """

    #  # load the model and label binarizer
    #  print("[INFO] loading network and label binarizer...")
    #  model = load_model(model)
    #  lb = pickle.loads(open(label_bin, "rb").read())

    lab2i = {label: j for j, label in enumerate(lb.classes_)}
    print(f"Lab2i {lab2i}")

    # make a prediction on the image
    preds = model.predict(all_images)
    print(f"Shape preds {preds.shape}")
    #  print(f'Preds {preds}')

    all_best_i = preds.argmax(axis=1)
    print(f"Shape all_best_i {all_best_i.shape}")

    confusion = np.zeros((len(lb.classes_), len(lb.classes_)), dtype=np.uint16)

    for j, pro in enumerate(preds):
        #  i = pro.argmax(axis=1)
        i = pro.argmax(axis=0)
        predicted_label = lb.classes_[i]
        correct = "TRUE" if true_labels[j] == predicted_label else "FALSE"
        print(
            f"True: {true_labels[j]}\tPredicted {predicted_label} with {pro[i]*100:.4f}%\t{correct}"
        )

        confusion[lab2i[predicted_label], lab2i[true_labels[j]]] += 1

    #  print(f'Confusion matrix\n{confusion}')
    return confusion, lb.classes_


def analyze_confusion(confusion, true_labels):
    """Compute the F-score from the confusion matrix, and print the intermediate results
    """
    print("Confusion matrix:")
    printer("Pre\Tru", true_labels)

    for line, label in zip(confusion, true_labels):
        printer(f"{label}", line)

    TP = confusion.diagonal()
    FN = np.sum(confusion, axis=0) - TP
    FP = np.sum(confusion, axis=1) - TP

    print()
    printer("TP", TP)
    printer("FP", FP)
    printer("FN", FN)

    # https://stackoverflow.com/a/37977222
    #  P = TP / ( TP + FP)
    #  R = TP / ( TP + FN)
    dP = TP + FP
    P = np.divide(TP, dP, out=np.zeros_like(TP, dtype=float), where=dP != 0)
    dR = TP + FN
    R = np.divide(TP, dR, out=np.zeros_like(TP, dtype=float), where=dR != 0)

    print("\nPrecision = TP / ( TP + FP)\tRecall = TP / ( TP + FN)")
    printer("Prec", P, ":.4f")
    printer("Recall", R, ":.4f")

    avgP = np.sum(P) / len(true_labels)
    avgR = np.sum(R) / len(true_labels)
    print(f"Average P: {avgP:.4f}\tR: {avgR:.4f}")

    print("F-score = 2 (PxR) / (P+R)")
    #  F = 2 (PxR) / (P+R)
    PdR = 2 * P * R
    PpR = P + R
    F = np.divide(PdR, PpR, out=np.zeros_like(TP, dtype=float), where=PpR != 0)
    printer("F-score", F, ":.4f")

    avgF = np.sum(F) / len(true_labels)
    print(f"Average F-score {avgF}")

    return avgF


def printer(header, iterable, formatter=""):
    row = header
    for item in iterable:
        #  row += f'\t{item{formatter}}'
        row += "\t{{i{f}}}".format(f=formatter).format(i=item)
    print(row)


def parse_args():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "-p",
        "--path-test",
        required=True,
        help="path to folder of input images we are going to classify",
    )

    ap.add_argument(
        "-b",
        "--basename",
        required=True,
        type=str,
        default="output/smallvggnet_test",
        help="basename for model and labels to use: {path/to/basename}.model",
    )

    ap.add_argument(
        "-w", "--width", type=int, default=64, help="target spatial dimension width"
    )
    ap.add_argument(
        "-e", "--height", type=int, default=64, help="target spatial dimension height"
    )

    args = ap.parse_args()
    return args


def main():
    # run with
    # python3 evaluate_model.py -w 64 -e 64 --model output/smallvggnet_full.model --label-bin output/smallvggnet_full.pickle --path-test kaa-test/

    args = parse_args()

    basename = args.basename
    model_path = f"{basename}.model"
    label_bin_path = f"{basename}.pickle"
    width = args.width
    height = args.height
    path_test = args.path_test

    all_images, true_labels = load_dataset(path_test, width, height)
    print(f"Shape all_images {all_images.shape}")
    #  cv2.imshow("Image resized", all_images[0])
    #  cv2.waitKey(0)

    # from path to object
    model, label_bin = load_model_label(model_path, label_bin_path)

    confusion, labels = compute_confusion_matrix(
        model, label_bin, all_images, true_labels
    )

    fscore = analyze_confusion(confusion, labels)


if __name__ == "__main__":
    main()

