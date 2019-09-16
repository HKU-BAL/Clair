import clair_model as cv
import utils
import matplotlib.pyplot as plt
import sys
import os
import argparse
import math
import numpy as np
import param
import matplotlib
matplotlib.use('Agg')


def Prepare(args):
    utils.setup_environment()


def PlotTensor(ofn, XArray):
    plot = plt.figure(figsize=(15, 8))

    plt.subplot(4, 1, 1)
    plt.xticks(np.arange(0, 33, 1))
    plt.yticks(np.arange(0, 8, 1), ['A', 'C', 'G', 'T', 'a', 'c', 'g', 't'])
    plt.imshow(XArray[0, :, :, 0].transpose(), vmin=0, vmax=50, interpolation="nearest", cmap=plt.cm.hot)
    plt.colorbar()

    plt.subplot(4, 1, 2)
    plt.xticks(np.arange(0, 33, 1))
    plt.yticks(np.arange(0, 8, 1), ['A', 'C', 'G', 'T', 'a', 'c', 'g', 't'])
    plt.imshow(XArray[0, :, :, 1].transpose(), vmin=-50, vmax=50, interpolation="nearest", cmap=plt.cm.bwr)
    plt.colorbar()

    plt.subplot(4, 1, 3)
    plt.xticks(np.arange(0, 33, 1))
    plt.yticks(np.arange(0, 8, 1), ['A', 'C', 'G', 'T', 'a', 'c', 'g', 't'])
    plt.imshow(XArray[0, :, :, 2].transpose(), vmin=-50, vmax=50, interpolation="nearest", cmap=plt.cm.bwr)
    plt.colorbar()

    plt.subplot(4, 1, 4)
    plt.xticks(np.arange(0, 33, 1))
    plt.yticks(np.arange(0, 8, 1), ['A', 'C', 'G', 'T', 'a', 'c', 'g', 't'])
    plt.imshow(XArray[0, :, :, 3].transpose(), vmin=-50, vmax=50, interpolation="nearest", cmap=plt.cm.bwr)
    plt.colorbar()

    plot.savefig(ofn, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(plot)


def CreatePNGs(args):
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

    XArray = np.array(splitted_array).reshape((-1, 33, 8, 4))
    _YArray = np.zeros((1, 16))
    varName = args.name
    print >> sys.stderr, "Plotting %s..." % (varName)

    # Create folder
    if not os.path.exists(varName):
        os.makedirs(varName)

    # Plot tensors
    PlotTensor(varName+"/tensor.png", XArray)


def ParseArgs():
    parser = argparse.ArgumentParser(
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


if __name__ == "__main__":
    args = ParseArgs()
    Prepare(args)
    CreatePNGs(args)
