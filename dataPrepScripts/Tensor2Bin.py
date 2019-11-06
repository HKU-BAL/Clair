import sys
import logging
import pickle
from argparse import ArgumentParser

import clair.utils as utils

logging.basicConfig(format='%(message)s', level=logging.INFO)


def pickle_dump(obj, file):
    return pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def Run(args):
    utils.setup_environment()

    logging.info("Loading the dataset ...")
    total, XArrayCompressed, YArrayCompressed, posArrayCompressed = \
        utils.get_training_array(
            tensor_fn=args.tensor_fn,
            var_fn=args.var_fn,
            bed_fn=args.bed_fn,
            shuffle=args.shuffle,
            is_allow_duplicate_chr_pos=args.allow_duplicate_chr_pos
        )

    logging.info("Writing to binary ...")
    with open(args.bin_fn, 'wb') as fh:
        pickle_dump(total, fh)
        pickle_dump(XArrayCompressed, fh)
        pickle_dump(YArrayCompressed, fh)
        pickle_dump(posArrayCompressed, fh)


def main():
    parser = ArgumentParser(description="Generate a binary format input tensor")

    parser.add_argument('--tensor_fn', type=str, default="vartensors",
                        help="Tensor input")

    parser.add_argument('--var_fn', type=str, default="truthvars",
                        help="Truth variants list input")

    parser.add_argument('--bed_fn', type=str, default=None,
                        help="High confident genome regions input in the BED format")

    parser.add_argument('--bin_fn', type=str, default=None,
                        help="Output a binary tensor file")

    parser.add_argument('--shuffle', action='store_true',
                        help="Shuffle on building bin")

    parser.add_argument('--allow_duplicate_chr_pos', action='store_true',
                        help="Allow duplicate chromosome:position in tensor input")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)


if __name__ == "__main__":
    main()
