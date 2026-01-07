from classifier.single_class_classifier import SingleClassClassifier
import argparse
import time


def main(args):
    base_dir = args["dir"]
    epochs = args["epochs"]

    start_time = time.time()

    classifier = SingleClassClassifier(base_dir)
    classifier.runTrainingAll(epochs=epochs)

    print("Total elapsed time: %s seconds" % (time.time() - start_time))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", default="/mnt/storage/ais/checkers/personal-care",
                    help="path to dcm root directory")
    ap.add_argument("--epochs", type=int, default=100, help="epochs")
    ap.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true")
    args = vars(ap.parse_args())

    main(args)
