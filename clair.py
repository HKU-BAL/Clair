#!/usr/bin/env python
import sys
from os.path import dirname, abspath
from importlib import import_module
from collections import namedtuple

from shared.param import REPO_NAME

DATA_PREP_SCRIPTS_FOLDER="dataPrepScripts"
DEEP_LEARNING_FOLDER="clair"
POST_PROCESS_SCRIPTS_FOLDER="clair.post_processing"

deep_learning_folder = [
    "callVarBamParallel",
    "callVarBam",
    "call_var",
    "evaluate",
    "plot_tensor",
    "train",
    "train_clr",
]
data_prep_scripts_folder = [
    "CreateTensor",
    "ExtractVariantCandidates",
    "GetTruth",
    "PairWithNonVariants",
    "Tensor2Bin",
    "CombineBins",
    "Bin2To3",
]
post_process_scripts_folder = [
    'ensemble',
    'overlap_variant',
]


def directory_for(submodule_name):
    if submodule_name in deep_learning_folder:
        return DEEP_LEARNING_FOLDER
    if submodule_name in data_prep_scripts_folder:
        return DATA_PREP_SCRIPTS_FOLDER
    if submodule_name in post_process_scripts_folder:
        return POST_PROCESS_SCRIPTS_FOLDER
    return ""


def print_help_messages():
    from textwrap import dedent
    print(dedent("""\
        {0} submodule invocator:
            Usage: python clair.py [submodule] [options of the submodule]

        Available data preparation submodules:\n{1}

        Available clair submodules:\n{2}

        Available post processing submodules:\n{3}
        """.format(
            REPO_NAME,
            "\n".join("          - %s" % submodule_name for submodule_name in data_prep_scripts_folder),
            "\n".join("          - %s" % submodule_name for submodule_name in deep_learning_folder),
            "\n".join("          - %s" % submodule_name for submodule_name in post_process_scripts_folder),
        )
    ))


def main():
    if len(sys.argv) <= 1 or sys.argv[1] == "-h" or sys.argv[1] == "--help":
        print_help_messages()
        sys.exit(0)

    submodule_name = sys.argv[1]
    if (
        submodule_name not in deep_learning_folder and
        submodule_name not in data_prep_scripts_folder and
        submodule_name not in post_process_scripts_folder
    ):
        sys.exit("[ERROR] Submodule %s not found." % (submodule_name))

    directory = directory_for(submodule_name)
    submodule = import_module("%s.%s" % (directory, submodule_name))

    # filter arguments (i.e. filter clair.py) and add ".py" for that submodule
    sys.argv = sys.argv[1:]
    sys.argv[0] += (".py")

    # Note: need to make sure every submodule contains main() method
    submodule.main()

    sys.exit(0)


if __name__ == "__main__":
    main()
