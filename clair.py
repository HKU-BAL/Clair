#!/usr/bin/env python

import os
import sys
import importlib
import subprocess

if sys.version_info.major >= 3:
    clv_path = os.path.dirname(os.path.abspath(__file__))+os.sep+'Clair'
    sys.path.insert(1, clv_path)


def mod(dir, name):
    if sys.argv[1] == name:
        sys.argv[0] = os.path.join(os.path.split(sys.argv[0])[0], dir, sys.argv[1] + ".py")
        del sys.argv[1]

        p = subprocess.Popen(["python"] + sys.argv)
        p.wait()

        sys.exit(0)


cl = ["callVarBamParallel", "callVarBam", "call_var", "evaluate",
      "plot_tensor", "tensor2Bin", "train_shuffle", "train", "validate"]
dp = ["ChooseItemInBed", "CombineMultipleDatasetsForTraining", "CountNumInBed", "CreateTensor",
      "ExtractVariantCandidates", "GetTruth", "PairWithNonVariants", "RandomSampling"]


def main():
    if len(sys.argv) <= 1:
        print ("Clair submodule invocator:")
        print ("  Usage: clair.py SubmoduleName [Options of the submodule]")
        print ("")
        print ("Available data preparation submodules:")
        for n in dp:
            print ("  - %s" % n)
        print ("")
        print ("Available clair submodules:")
        for n in cl:
            print ("  - %s" % n)
        print ("")
        print ("Data preparation scripts:")
        print ("%s/dataPrepScripts" % os.path.dirname(os.path.abspath(sys.argv[0])))
        print ("")
        print ("Clair scripts:")
        print ("%s/Clair" % os.path.dirname(os.path.abspath(sys.argv[0])))
        print ("")
        sys.exit(0)

    for n in cl:
        mod("Clair", n)
    for n in dp:
        mod("dataPrepScripts", n)


if __name__ == "__main__":
    main()
