import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from argparse import ArgumentParser

from clair.utils import setup_environment

def plot_tensor(ofn, XArray):
    plot = plt.figure(figsize=(15, 8))

    plot_min = -30
    plot_max = 30
    plot_arr = ["A+", "C+", "G+", "T+", "A-", "C-", "G-", "T-"]

    plt.subplot(4, 1, 1)
    plt.xticks(np.arange(0, 33, 1))
    plt.yticks(np.arange(0, 8, 1), plot_arr)
    plt.imshow(XArray[0, :, :, 0].transpose(), vmin=0, vmax=plot_max, interpolation="nearest", cmap=plt.cm.hot)
    plt.colorbar()

    plt.subplot(4, 1, 2)
    plt.xticks(np.arange(0, 33, 1))
    plt.yticks(np.arange(0, 8, 1), plot_arr)
    plt.imshow(XArray[0, :, :, 1].transpose(), vmin=plot_min, vmax=plot_max, interpolation="nearest", cmap=plt.cm.bwr)
    plt.colorbar()

    plt.subplot(4, 1, 3)
    plt.xticks(np.arange(0, 33, 1))
    plt.yticks(np.arange(0, 8, 1), plot_arr)
    plt.imshow(XArray[0, :, :, 2].transpose(), vmin=plot_min, vmax=plot_max, interpolation="nearest", cmap=plt.cm.bwr)
    plt.colorbar()

    plt.subplot(4, 1, 4)
    plt.xticks(np.arange(0, 33, 1))
    plt.yticks(np.arange(0, 8, 1), plot_arr)
    plt.imshow(XArray[0, :, :, 3].transpose(), vmin=plot_min, vmax=plot_max, interpolation="nearest", cmap=plt.cm.bwr)
    plt.colorbar()

    plot.savefig(ofn, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(plot)


def create_png(args):
    f = open(args.array_fn, 'r')
    array = f.read()
    f.close()
    import re
    array = re.split("\n", array)
    array = [x for x in array if x]
    print(array)

    splitted_array = []
    for i in range(len(array)):
        splitted_array += re.split(",", array[i])

    print("splitted array length")
    print(len(splitted_array))
    print(splitted_array[0])
    # for i in range(len(splitted_array)):
    #     splitted_array[i] = int(splitted_array[i])

    XArray = np.array(splitted_array, dtype=np.float32).reshape((-1, 33, 8, 4))
    XArray[0, :, :, 1] -= XArray[0, :, :, 0]
    XArray[0, :, :, 2] -= XArray[0, :, :, 0]
    XArray[0, :, :, 3] -= XArray[0, :, :, 0]

    _YArray = np.zeros((1, 16))
    varName = args.name
    print("Plotting %s..." % (varName), file=sys.stderr)

    # Create folder
    if not os.path.exists(varName):
        os.makedirs(varName)

    # Plot tensors
    plot_tensor(varName+"/tensor.png", XArray)


def ParseArgs():
    parser = ArgumentParser(
        description="Visualize tensors and hidden layers in PNG")

    parser.add_argument('--array_fn', type=str, default="vartensors",
                        help="Array input")

    parser.add_argument('--name', type=str, default=None,
                        help="output name")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    return args


def main():
    args = ParseArgs()
    setup_environment()
    create_png(args)


if __name__ == "__main__":
    main()
