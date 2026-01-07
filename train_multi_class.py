from classifier.multi_class_classifier import MultiClassClassifier
import argparse
import time


def main(args):
    base_dir = args["dir"]
    epochs = args["epochs"]
    is_mapped = args["map"]

    start_time = time.time()

    classifier = MultiClassClassifier(base_dir)
    classifier.runTraining(epochs=epochs, is_mapped=is_mapped)

    print("Total elapsed time: %s seconds" % (time.time() - start_time))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", default="../data",
                    help="path to dcm root directory")
    ap.add_argument("--epochs", type=int, default=100, help="epochs")
    ap.add_argument("--map", help="training data mapped flag",
                    action='store_true')
    ap.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true")
    args = vars(ap.parse_args())

    main(args)
