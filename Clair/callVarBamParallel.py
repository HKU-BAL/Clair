import os
import sys
import subprocess
import intervaltree
import shlex
import argparse
import param
from collections import namedtuple

majorContigs = {"chr"+str(a) for a in range(0, 23)+["X", "Y"]}.union({str(a) for a in range(0, 23)+["X", "Y"]})

CommandOption = namedtuple('CommandOption', ['option', 'value'])
CommandOptionWithNoValue = namedtuple('CommandOptionWithNoValue', ['option'])
ExecuteCommand = namedtuple('ExecuteCommand', ['bin', 'bin_value'])


def command_string_from(command):
    if isinstance(command, CommandOption):
        return "--{} \"{}\"".format(command.option, command.value)
    elif isinstance(command, CommandOptionWithNoValue):
        return "--{}".format(command.option)
    elif isinstance(command, ExecuteCommand):
        return " ".join([command.bin, command.bin_value])
    else:
        return ""


def executable_command_string_from(commands):
    return " ".join(map(command_string_from, commands))


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


def intervaltree_from(bed_file_path):
    tree = {}
    if bed_file_path is None:
        return tree

    bed_fp = subprocess.Popen(shlex.split("pigz -fdc %s" % (bed_file_path)), stdout=subprocess.PIPE, bufsize=8388608)
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

    return tree


def Run(args):
    basedir = os.path.dirname(__file__)
    if len(basedir) == 0:
        callVarBamBin = CheckFileExist("./callVarBam.py")
    else:
        callVarBamBin = CheckFileExist(basedir + "/callVarBam.py")

    pypyBin = CheckCmdExist(args.pypy)
    samtoolsBin = CheckCmdExist(args.samtools)
    chkpnt_fn = CheckFileExist(args.chkpnt_fn, sfx=".meta")
    bam_fn = CheckFileExist(args.bam_fn)
    ref_fn = CheckFileExist(args.ref_fn)
    fai_fn = CheckFileExist(args.ref_fn + ".fai")
    bed_fn = CheckFileExist(args.bed_fn) if args.bed_fn is not None else None
    output_prefix = args.output_prefix
    threshold = args.threshold

    is_bed_file_provided = bed_fn is not None

    minCoverage = args.minCoverage
    sampleName = args.sampleName
    delay = args.delay
    threads = args.tensorflowThreads
    qual = args.qual
    includingAllContigs = args.includingAllContigs
    refChunkSize = args.refChunkSize

    required_commands = [
        ExecuteCommand('python', callVarBamBin),
        CommandOption('chkpnt_fn', chkpnt_fn),
        CommandOption('ref_fn', ref_fn),
        CommandOption('bam_fn', bam_fn),
        CommandOption('threshold', threshold),
        CommandOption('minCoverage', minCoverage),
        CommandOption('pypy', pypyBin),
        CommandOption('samtools', samtoolsBin),
        CommandOption('delay', delay),
        CommandOption('threads', threads),
        CommandOption('sampleName', sampleName),
    ]

    optional_options = []
    activation_only_commands= []
    vcf_fn = CheckFileExist(args.vcf_fn) if args.vcf_fn else None
    if vcf_fn is not None:
        optional_options.append(CommandOption('vcf_fn', vcf_fn))
    if args.qual is not None:
        optional_options.append(CommandOption('qual', qual))
    if args.considerleftedge:
        optional_options.append(CommandOptionWithNoValue('considerleftedge'))
    if args.debug:
        optional_options.append(CommandOptionWithNoValue('debug'))
    if args.pysam_for_all_indel_bases:
        optional_options.append(CommandOptionWithNoValue('pysam_for_all_indel_bases'))
    if args.activation_only:
        activation_only_commands.append(CommandOptionWithNoValue('activation_only'))
        if args.log_path is not None:
            activation_only_commands.append(CommandOption('log_path', args.log_path))
        if args.max_plot is not None:
            activation_only_commands.append(CommandOption('max_plot', args.max_plot))
        if args.parallel_level is not None:
            activation_only_commands.append(CommandOption('parallel_level', args.parallel_level))
        if args.workers is not None:
            activation_only_commands.append(CommandOption('workers', args.workers))
        if args.fast_plotting:
            activation_only_commands.append(CommandOptionWithNoValue('fast_plotting'))

    command_string = executable_command_string_from(required_commands + optional_options)

    tree = intervaltree_from(bed_file_path=bed_fn)

    with open(fai_fn, 'r') as fai_fp:
        for line in fai_fp:
            fields = line.strip().split("\t")

            chromName = fields[0]
            if includingAllContigs == False and str(chromName) not in majorContigs:
                continue
            regionStart = 0
            chromLength = int(fields[1])

            while regionStart < chromLength:
                start = regionStart
                end = regionStart + refChunkSize
                if end > chromLength:
                    end = chromLength
                output_fn = "%s.%s_%d_%d.vcf" % (output_prefix, chromName, start, end)
                regionStart = end

                additional_options = [
                    CommandOption('ctgName', chromName),
                    CommandOption('ctgStart', start),
                    CommandOption('ctgEnd', end),
                    CommandOption('call_fn', output_fn)
                ]

                if not is_bed_file_provided:
                    print(command_string + " " + executable_command_string_from(
                            additional_options + activation_only_commands if args.activation_only else []))

                if chromName in tree and len(tree[chromName].search(start, end)) != 0:
                    additional_options.append(CommandOption('bed_fn', bed_fn))
                    if args.activation_only:
                        print(command_string + " " + executable_command_string_from(additional_options+activation_only_commands))
                    else:
                        print(command_string + " " + executable_command_string_from(
                                    additional_options))



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

    parser.add_argument('--includingAllContigs', action='store_true',
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

    parser.add_argument('--debug', action='store_true',
                        help="Debug mode, optional")

    parser.add_argument('--pysam_for_all_indel_bases', action='store_true',
                        help="Always using pysam for outputting indel bases, optional")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)
