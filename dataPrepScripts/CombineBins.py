import os
import pickle
import argparse
import random
from collections import namedtuple

Data = namedtuple('Data', ['x', 'y', 'pos', 'total'])


def process_command():
    parser = argparse.ArgumentParser(description="Combine small bins into a large bin.")
    parser.add_argument(
        '--src', type=str, default=os.path.join(os.curdir, "all_bins"),
        help="Path to directory that stores small bins. (default: %(default)s)"
    )
    parser.add_argument(
        '--dst', type=str, default=os.curdir,
        help="Path of the output folder. (default: %(default)s)"
    )
    parser.add_argument(
        '--bin_name', type=str, default="tensor.bin",
        help="Name of the large bin. (default: %(default)s)"
    )
    parser.add_argument(
        '--shuffle_data', type=bool, default=False,
        help="Shuffle data after loaded all data. (default: %(default)s)"
    )

    return parser.parse_args()


def load_data_from(directory_path, need_shuffle_file_paths=False):
    X = []
    Y = []
    pos = []
    total = 0

    file_paths = os.listdir(directory_path)
    if need_shuffle_file_paths:
        random.shuffle(file_paths)

    for file_path in file_paths:
        absolute_file_path = os.path.abspath(os.path.join(directory_path, file_path))
        print "[INFO] Load data from: {}".format(absolute_file_path)
        with open(absolute_file_path, "rb") as f:
            a = int(pickle.load(f))
            x = pickle.load(f)
            y = pickle.load(f)
            p = pickle.load(f)

            total += a
            X += x
            Y += y
            pos += p

    return Data(x=X, y=Y, pos=pos, total=total)


def output_data(dst, data):
    print "[INFO] Output: {}".format(os.path.abspath(dst))
    with open(dst, "wb") as f:
        pickle.dump(data.total, f)
        pickle.dump(data.x, f)
        pickle.dump(data.y, f)
        pickle.dump(data.pos, f)


if __name__ == "__main__":
    args = process_command()

    data = load_data_from(
        directory_path=args.src,
        need_shuffle_file_paths=args.shuffle_data
    )

    output_data(
        dst=os.path.join(args.dst, args.bin_name),
        data=data
    )
