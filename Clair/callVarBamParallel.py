import os
import sys
import subprocess
import intervaltree
import shlex
import argparse
import param

majorContigs = {"chr"+str(a) for a in range(0, 23)+["X", "Y"]}.union({str(a) for a in range(0, 23)+["X", "Y"]})


def CheckFileExist(fn, sfx=""):
    if not os.path.isfile(fn+sfx):
        sys.exit("Error: %s not found" % (fn+sfx))
    return os.path.abspath(fn)


def CheckCmdExist(cmd):
    try:
        subprocess.check_output("which %s" % (cmd), shell=True)
    except:
        sys.exit("Error: %s executable not found" % (cmd))
    return cmd


def Run(args):
    basedir = os.path.dirname(__file__)
    prefix_1=[]
    suffix_1=[]
    if len(basedir) == 0:
        callVarBamBin = CheckFileExist("./callVarBam.py")
    else:
        callVarBamBin = CheckFileExist(basedir + "/callVarBam.py")
    prefix_1.append("python %s" %(callVarBamBin))
    chkpnt_fn = CheckFileExist(args.chkpnt_fn, sfx=".meta")
    prefix_1.append("--chkpnt_fn %s" % (chkpnt_fn))
    ref_fn = CheckFileExist(args.ref_fn)
    prefix_1.append("--ref_fn %s" % (ref_fn))
    bam_fn = CheckFileExist(args.bam_fn)
    prefix_1.append("--bam_fn %s" % (bam_fn))
    bed_fn = CheckFileExist(args.bed_fn) if args.bed_fn != None else None
    prefix_1.append("--bed_fn %s" % (bed_fn))
    threshold = args.threshold
    prefix_1.append("--threshold %f" % (threshold))
    minCoverage = args.minCoverage
    prefix_1.append("--minCoverage %f" % (minCoverage))
    pypyBin = CheckCmdExist(args.pypy)
    prefix_1.append("--pypy %s" % (pypyBin))
    samtoolsBin = CheckCmdExist(args.samtools)
    prefix_1.append("--samtools %s" % (samtoolsBin))
    delay = args.delay
    prefix_1.append("--delay %d" % (delay))
    threads = args.tensorflowThreads
    prefix_1.append("--threads %d" % (threads))
    sampleName = args.sampleName
    prefix_1.append("--sampleName %s" % (sampleName))
    vcf_fn = "--vcf_fn %s" % (CheckFileExist(args.vcf_fn)) if args.vcf_fn != None else ""
    prefix_1.append("%s" %(vcf_fn))
    considerleftedge = "--considerleftedge" if args.considerleftedge else ""
    prefix_1.append("%s" % (considerleftedge))
    log_path="--log_path {}".format(args.log_path) if args.log_path else ""
    qual = "--qual %d" % (args.qual) if args.qual else ""
    fast_plotting= "--fast_plotting" if args.fast_plotting else ""
    debug = "--debug" if args.debug else ""
    suffix_1.append("--activation_only %s --max_plot %d --parallel_level %d --workers %d %s %s %s" %\
              (log_path, args.max_plot, args.parallel_level, args.workers, qual, fast_plotting, debug))
    suffix_2="%s %s" % (qual, debug)
    fai_fn = CheckFileExist(args.ref_fn + ".fai")
    output_prefix = args.output_prefix

    includingAllContigs = args.includingAllContigs
    refChunkSize = args.refChunkSize

    tree = {}
    if bed_fn != None:
        bed_fp = subprocess.Popen(shlex.split("pigz -fdc %s" % (bed_fn)), stdout=subprocess.PIPE, bufsize=8388608)
        for row in bed_fp.stdout:
            row = row.strip().split()
            name = row[0]
            if name not in tree:
                tree[name] = intervaltree.IntervalTree()
            begin = int(row[1])
            end = int(row[2])-1
            if end == begin:
                end += 1
            tree[name].addi(begin, end)
        bed_fp.stdout.close()
        bed_fp.wait()

    fai_fp = open(fai_fn)
    for line in fai_fp:

        fields = line.strip().split("\t")

        chromName = fields[0]
        prefix_1.insert(5,"--ctgName %s" % (chromName))
        if includingAllContigs == False and str(chromName) not in majorContigs:
            continue
        regionStart = 0
        prefix_1.insert(6, "--ctgStart %d" % (regionStart))
        chromLength = int(fields[1])

        while regionStart < chromLength:
            start = regionStart
            end = regionStart + refChunkSize
            prefix_1.insert(7, "--ctgEnd %d" % (end))
            if end > chromLength:
                end = chromLength
            output_fn = "%s.%s_%d_%d.vcf" % (output_prefix, chromName, regionStart, end)
            prefix_1.insert(8,"--call_fn %s" % (output_fn))
            prefix_2=prefix_1
            prefix_2.pop(4)
            if bed_fn != None and chromName in tree and len(tree[chromName].search(start, end)) != 0:
                    if args.activation_only:
                        print(" ".join(prefix_1+suffix_1))
                    else:
                        print(" ".join(prefix_1+suffix_2))
            elif args.activation_only:
                print(" ".join(prefix_2+suffix_1))
            else:
                print(" ".join(prefix_2+suffix_2))
            regionStart = end
            prefix_1.pop(8)
            prefix_1.pop(7)
        prefix_1.pop(6)
        prefix_1.pop(5)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create commands for calling variants in parallel using a trained Clair model and a BAM file")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a Clair model")

    parser.add_argument('--ref_fn', type=str, default="ref.fa",
                        help="Reference fasta file input, default: %(default)s")

    parser.add_argument('--bed_fn', type=str, default=None,
                        help="Call variant only in these regions, optional, default: whole genome")

    parser.add_argument('--refChunkSize', type=int, default=10000000,
                        help="Divide job with smaller genome chunk size for parallelism, default: %(default)s")

    parser.add_argument('--bam_fn', type=str, default="bam.bam",
                        help="BAM file input, default: %(default)s")

    parser.add_argument('--vcf_fn', type=str, default=None,
                        help="Candidate sites VCF file input, if provided, variants will only be called at the sites in the VCF file,  default: %(default)s")

    parser.add_argument('--output_prefix', type=str, default=None,
                        help="Output prefix")

    parser.add_argument('--includingAllContigs', type=param.str2bool, nargs='?', const=True, default=False,
                        help="Call variants on all contigs, default: chr{1..22,X,Y,M,MT} and {1..22,X,Y,MT}")

    parser.add_argument('--tensorflowThreads', type=int, default=4,
                        help="Number of threads per tensorflow job, default: %(default)s")

    parser.add_argument('--threshold', type=float, default=0.2,
                        help="Minimum allele frequence of the 1st non-reference allele for a site to be considered as a condidate site, default: %(default)f")

    parser.add_argument('--minCoverage', type=float, default=4,
                        help="Minimum coverage required to call a variant, default: %(default)d")

    parser.add_argument('--qual', type=int, default=None,
                        help="If set, variant with equal or higher quality will be marked PASS, or LowQual otherwise, optional")

    parser.add_argument('--sampleName', type=str, default="SAMPLE",
                        help="Define the sample name to be shown in the VCF file")

    parser.add_argument('--considerleftedge', type=param.str2bool, nargs='?', const=True, default=True,
                        help="Count the left-most base-pairs of a read for coverage even if the starting position of a read is after the starting position of a tensor, default: %(default)s")

    parser.add_argument('--samtools', type=str, default="samtools",
                        help="Path to the 'samtools', default: %(default)s")

    parser.add_argument('--pypy', type=str, default="pypy",
                        help="Path to the 'pypy', default: %(default)s")

    parser.add_argument('--delay', type=int, default=10,
                        help="Wait a short while for no more than %(default)s to start the job. This is to avoid starting multiple jobs simultaneously that might use up the maximum number of threads allowed, because Tensorflow will create more threads than needed at the beginning of running the program.")

    parser.add_argument('--activation_only', action='store_true',
                        help="Output activation only, no prediction")

    parser.add_argument('--max_plot', type=int, default=10,
                        help="The maximum number of plots output, negative number means no limit (plot all), default: %(default)s")

    parser.add_argument('--log_path', type=str, nargs='?', default=None,
                        help="The path for tensorflow logging, default: %(default)s")

    parser.add_argument('-p', '--parallel_level', type=int, default=2,
                        help="The level of parallelism in plotting (currently available: 0, 2), default: %(default)s")

    parser.add_argument('-w', '--workers', type=int, default=8,
                        help="The number of workers in plotting, default: %(default)s")

    parser.add_argument('--fast_plotting', action='store_true',
                        help="Enable fast plotting.")

    parser.add_argument('--debug', type=param.str2bool, nargs='?', const=True, default=False,
                        help="Debug mode, optional")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)
