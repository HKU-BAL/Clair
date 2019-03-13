import sys
import argparse
import subprocess
import shlex
import os


class TruthStdout(object):
    def __init__(self, handle):
        self.stdin = handle

    def __del__(self):
        self.stdin.close()


def CheckFileExist(fn):
    if not os.path.isfile(fn):
        return None
    return os.path.abspath(fn)


def CheckCmdExist(cmd):
    try:
        subprocess.check_output("which %s" % (cmd), shell=True)
    except:
        return None
    return cmd


def OutputVariant(args):
    var_fn = args.var_fn
    vcf_fn = args.vcf_fn
    ctgName = args.ctgName
    ctgStart = args.ctgStart
    ctgEnd = args.ctgEnd
    if ctgStart != None and ctgEnd != None:
        ctgStart += 1

    if args.var_fn != "PIPE":
        var_fpo = open(var_fn, "wb")
        var_fp = subprocess.Popen(shlex.split("gzip -c"), stdin=subprocess.PIPE,
                                  stdout=var_fpo, stderr=sys.stderr, bufsize=8388608)
    else:
        var_fp = TruthStdout(sys.stdout)

    tabixed = 0
    if ctgStart != None and ctgEnd != None:
        if CheckFileExist("%s.tbi" % (vcf_fn)) != None:
            if CheckCmdExist("tabix") != None:
                tabixed = 1
                vcf_fp = subprocess.Popen(shlex.split("tabix -f -p vcf %s %s:%s-%s" %
                                                      (vcf_fn, ctgName, ctgStart, ctgEnd)), stdout=subprocess.PIPE, bufsize=8388608)
    if tabixed == 0:
        vcf_fp = subprocess.Popen(shlex.split("gzip -fdc %s" % (vcf_fn)), stdout=subprocess.PIPE, bufsize=8388608)
    for row in vcf_fp.stdout:
        row = row.strip().split()
        if row[0][0] == "#":
            continue
        if row[0] != ctgName:
            continue
        if ctgStart != None and ctgEnd != None:
            if int(row[1]) < ctgStart or int(row[1]) > ctgEnd:
                continue
        last = row[-1]

        # normal GetTruth
        varType = last.split(":")[0].replace("/", "|").replace(".", "0").split("|")
        p1, p2 = varType

        # 1000 Genome GetTruth (format problem)
        # p1, p2 = "1", "1"

        chromosome = row[0]
        position = row[1]
        reference = row[3]
        alternate = row[4]

        genotype_1, genotype_2 = (p1, p2) if int(p1) < int(p2) else (p2, p1)

        var_fp.stdin.write(" ".join([chromosome, position, reference, alternate, genotype_1, genotype_2]))
        var_fp.stdin.write("\n")

    vcf_fp.stdout.close()
    vcf_fp.wait()

    if args.var_fn != "PIPE":
        var_fp.stdin.close()
        var_fp.wait()
        var_fpo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract variant type and allele from a Truth dataset")

    parser.add_argument('--vcf_fn', type=str, default="input.vcf",
                        help="Truth vcf file input, default: %(default)s")

    parser.add_argument('--var_fn', type=str, default="PIPE",
                        help="Truth variants output, use PIPE for standard output, default: %(default)s")

    parser.add_argument('--ctgName', type=str, default="chr17",
                        help="The name of sequence to be processed, default: %(default)s")

    parser.add_argument('--ctgStart', type=int, default=None,
                        help="The 1-bsae starting position of the sequence to be processed")

    parser.add_argument('--ctgEnd', type=int, default=None,
                        help="The inclusive ending position of the sequence to be processed")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    OutputVariant(args)
