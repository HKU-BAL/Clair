import sys
import shlex
from subprocess import PIPE
from argparse import ArgumentParser
from collections import namedtuple

from shared.utils import file_path_from, executable_command_string_from, subprocess_popen

VariantInfo = namedtuple('VariantInfo', ['chromosome', 'position', 'reference', 'alternate', 'genotype_1', 'genotype_2'])

class TruthStdout(object):
    def __init__(self, handle):
        self.stdin = handle

    def __del__(self):
        self.stdin.close()

def GetBase(chromosome, position, ref_fn):
    fp = subprocess_popen(shlex.split("samtools faidx %s %s:%s-%s" % (ref_fn, chromosome, position, position)))
    for line in fp.stdout:
        if line[0] == ">":
            continue
        else:
            return line.strip()

def GetLineFromInfo(variant_info):
    return (" ".join(variant_info) + "\n")

def GetInfosFromVar(variant_info, ref_fn):
    chromosome, position, reference, alternate, genotype_1, genotype_2 = variant_info
    if "*" not in alternate:
        return [variant_info]
    else:
        if ref_fn is None:
            sys.exit("Please provide a reference file correspond to the vcf.")
        try:
            alternate_list = alternate.split(",")
        except e:
            print(e, file=sys.stderr)
            sys.exit("Exception occured when getting true variant, exiting ...")

        if alternate_list[1] == "*":
            alternate_list[0], alternate_list[1] = alternate_list[1], alternate[0]

        lines = []
        for alt in alternate_list:
            if alt == "*":
                new_pos = str(int(position)-1)
                new_alt = GetBase(chromosome, new_pos, ref_fn)
                new_ref = new_alt + reference[0]
                lines.append(VariantInfo(chromosome, new_pos, new_ref, new_alt, "0", "1"))
            else:
                lines.append(VariantInfo(chromosome, position, reference, alt, "0", "1"))

        return lines

def MergeInfos(info_1, info_2):
    if "," in info_1.reference or "," in info_1.alternate:
        return info_1
    if info_1.reference == info_2.reference:
        if info_1.alternate == info_2.alternate:
            return info_1
        else:
            new_alternate = "{},{}".format(info_1.alternate, info_2.alternate)
            return VariantInfo(info_1.chromosome, info_1.position, info_1.reference, new_alternate, "1", "2")
    else:
        if len(info_1.alternate) > len(info_2.alternate):
            info_1, info_2 = info_2, info_1
        new_ref = info_2.reference
        new_alternate = "{},{}".format(info_1.alternate + info_2.reference[len(info_1.reference)-len(info_2.reference):], info_2.alternate)
        return VariantInfo(info_1.chromosome, info_1.position, new_ref, new_alternate, "1", "2")

def OutputVariant(args):
    var_fn = args.var_fn
    vcf_fn = args.vcf_fn
    ref_fn = args.ref_fn
    ctg_name = args.ctgName
    ctg_start = args.ctgStart
    ctg_end = args.ctgEnd

    if args.var_fn != "PIPE":
        var_fpo = open(var_fn, "wb")
        var_fp = subprocess_popen(shlex.split("gzip -c"), stdin=PIPE, stdout=var_fpo)
    else:
        var_fp = TruthStdout(sys.stdout)

    is_ctg_region_provided = ctg_start is not None and ctg_end is not None
    if (
        is_ctg_region_provided and
        file_path_from("%s.tbi" % (vcf_fn)) is not None and
        executable_command_string_from("tabix") is not None
    ):
        vcf_fp = subprocess_popen(shlex.split("tabix -f -p vcf %s %s:%s-%s" % (vcf_fn, ctg_name, ctg_start, ctg_end)))
    else:
        vcf_fp = subprocess_popen(shlex.split("gzip -fdc %s" % (vcf_fn)))

    buffer_line = None
    buffer_line_pos = -1

    for row in vcf_fp.stdout:
        columns = row.strip().split()
        if columns[0][0] == "#":
            continue

        # position in vcf is 1-based
        chromosome, position = columns[0], columns[1]
        if chromosome != ctg_name:
            continue
        if is_ctg_region_provided and not (ctg_start <= int(position) <= ctg_end):
            continue
        reference, alternate, last_column = columns[3], columns[4], columns[-1]

        # normal GetTruth
        genotype = last_column.split(":")[0].replace("/", "|").replace(".", "0").split("|")
        genotype_1, genotype_2 = genotype

        # 1000 Genome GetTruth (format problem) (no genotype is given)
        # genotype_1, genotype_2 = "1", "1"
        # if alternate.find(',') >= 0:
        #     genotype_1, genotype_2 = "1", "2"

        if int(genotype_1) > int(genotype_2):
            genotype_1, genotype_2 = genotype_2, genotype_1

        info_line = VariantInfo(chromosome, position, reference, alternate, genotype_1, genotype_2)

        for info in GetInfosFromVar(info_line, ref_fn):
            if int(info.position) == buffer_line_pos:
                buffer_line = MergeInfos(buffer_line, info)
            else:
                if buffer_line != None:
                    var_fp.stdin.write(GetLineFromInfo(buffer_line))
                buffer_line = info
                buffer_line_pos = int(buffer_line.position)

    var_fp.stdin.write(GetLineFromInfo(buffer_line))

    vcf_fp.stdout.close()
    vcf_fp.wait()

    if args.var_fn != "PIPE":
        var_fp.stdin.close()
        var_fp.wait()
        var_fpo.close()


def main():
    parser = ArgumentParser(description="Extract variant type and allele from a Truth dataset")

    parser.add_argument('--vcf_fn', type=str, default="input.vcf",
                        help="Truth vcf file input, default: %(default)s")

    parser.add_argument('--var_fn', type=str, default="PIPE",
                        help="Truth variants output, use PIPE for standard output, default: %(default)s")

    parser.add_argument('--ref_fn', type=str, default=None,
                        help="Reference file input, must be provided if the vcf contains '*' in ALT field.")

    parser.add_argument('--ctgName', type=str, default="chr17",
                        help="The name of sequence to be processed, default: %(default)s")

    parser.add_argument('--ctgStart', type=int, default=None,
                        help="The 1-based starting position of the sequence to be processed")

    parser.add_argument('--ctgEnd', type=int, default=None,
                        help="The 1-based inclusive ending position of the sequence to be processed")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    OutputVariant(args)


if __name__ == "__main__":
    main()
