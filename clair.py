#!/usr/bin/env python

import os
import sys
import importlib
import subprocess


clair_folder = [
    "callVarBamParallel",
    "callVarBam",
    "call_var",
    "evaluate",
    "plot_tensor",
    "tensor2Bin",
    "train",
    "train_clr",
]
data_prep_scripts_folder = [
    "CombineBins",
    "CreateTensor",
    "ExtractVariantCandidates",
    "GetTruth",
    "PairWithNonVariants",
]


def directory_for(submodule_name):
    if submodule_name in clair_folder:
        return "Clair"
    if submodule_name in data_prep_scripts_folder:
        return "dataPrepScripts"
    return ""


def print_help_messages():
    from textwrap import dedent
    print dedent("""\
        Clair submodule invocator:
            Usage: clair.py SubmoduleName [Options of the submodule]

        Available data preparation submodules:\n{0}

        Available clair submodules:\n{1}

        Data preparation scripts:
        {2}

        Clair scripts:
        {3}
        """.format(
            "\n".join("          - %s" % submodule_name for submodule_name in data_prep_scripts_folder),
            "\n".join("          - %s" % submodule_name for submodule_name in clair_folder),
            "%s/dataPrepScripts" % os.path.dirname(os.path.abspath(sys.argv[0])),
            "%s/Clair" % os.path.dirname(os.path.abspath(sys.argv[0]))
        )
    )


def main():
    if len(sys.argv) <= 1:
        print_help_messages()
        sys.exit(0)

    submodule_name = sys.argv[1]
    if (
        submodule_name not in clair_folder and
        submodule_name not in data_prep_scripts_folder
    ):
        sys.exit("[ERROR] Submodule %s not found." % (submodule_name))

    directory = directory_for(submodule_name)
    submodule = importlib.import_module("%s.%s" % (directory, submodule_name))

    # filter arguments (i.e. filter clair.py) and add ".py" for that submodule
    sys.argv = sys.argv[1:]
    sys.argv[0] += (".py")

    # Note: need to make sure every submodule contains main() method
    submodule.main()

    sys.exit(0)


if __name__ == "__main__":
    main()
