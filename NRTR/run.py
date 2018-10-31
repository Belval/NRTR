import argparse
from nrtr import NRTR

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Train or test the NRTR model.')

    parser.add_argument(
        "--train",
        action="store_true",
        help="Define if we train the model"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Define if we test the model"
    )
    parser.add_argument(
        "-ttr",
        "--train_test_ratio",
        type=float,
        nargs="?",
        help="How the data will be split between training and testing",
        default=0.70
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        nargs="?",
        help="The path where the pretrained model can be found or where the model will be saved",
        default='./save/'
    )
    parser.add_argument(
        "-ex",
        "--examples_path",
        type=str,
        nargs="?",
        help="The path to the file containing the examples (training samples)",
        required=True
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        nargs="?",
        help="Size of a batch",
        default=64
    )
    parser.add_argument(
        "-it",
        "--iteration_count",
        type=int,
        nargs="?",
        help="How many iteration in training",
        default=1000
    )
    parser.add_argument(
        "-miw",
        "--max_image_width",
        type=int,
        nargs="?",
        help="Maximum width of an example before truncating",
        default=100
    )

    parser.add_argument(
        "-r",
        "--restore",
        action="store_true",
        help="Define if we try to load a checkpoint file from the save folder"
    )

    return parser.parse_args()

def main():
    """
        Entry point when using NRTR from the commandline
    """

    args = parse_arguments()

    if not args.train and not args.test:
        print("If we are not training, and not testing, what is the point?")

    nrtr = None

    if args.train:
        nrtr = NRTR(
            args.batch_size,
            args.model_path,
            args.examples_path,
            args.max_image_width,
            args.train_test_ratio,
            args.restore
        )

        nrtr.train(args.iteration_count)

    if args.test:
        if nrtr is None:
            nrtr = NRTR(
                args.batch_size,
                args.model_path,
                args.examples_path,
                args.max_image_width,
                0,
                args.restore
            )

        nrtr.test()

if __name__ == '__main__':
    main()
