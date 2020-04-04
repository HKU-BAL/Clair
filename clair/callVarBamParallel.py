import os
import sys
import argparse

from shared.command_options import (
    CommandOption,
    CommandOptionWithNoValue,
    ExecuteCommand,
    command_string_from,
    command_option_from
)
from shared.interval_tree import bed_tree_from, is_region_in
from shared.utils import file_path_from, executable_command_string_from

major_contigs = {"chr"+str(a) for a in list(range(1, 23))+["X", "Y"]}.union({str(a) for a in list(range(1, 23))+["X", "Y"]})


def Run(args):
    basedir = os.path.dirname(__file__)

    callVarBamBin = basedir + "/../clair.py callVarBam"
    pypyBin = executable_command_string_from(args.pypy, exit_on_not_found=True)
    samtoolsBin = executable_command_string_from(args.samtools, exit_on_not_found=True)

    chkpnt_fn = file_path_from(args.chkpnt_fn, suffix=".meta", exit_on_not_found=True)
    bam_fn = file_path_from(args.bam_fn, exit_on_not_found=True)
    ref_fn = file_path_from(args.ref_fn, exit_on_not_found=True)
    fai_fn = file_path_from(args.ref_fn + ".fai", exit_on_not_found=True)
    bed_fn = file_path_from(args.bed_fn)
    vcf_fn = file_path_from(args.vcf_fn)

    output_prefix = args.output_prefix
    af_threshold = args.threshold

    tree = bed_tree_from(bed_file_path=bed_fn)

    minCoverage = args.minCoverage
    sampleName = args.sampleName
    delay = args.delay
    threads = args.tensorflowThreads
    qual = args.qual
    is_include_all_contigs = args.includingAllContigs
    region_chunk_size = args.refChunkSize

    stop_consider_left_edge = command_option_from(args.stop_consider_left_edge, 'stop_consider_left_edge')
    log_path = command_option_from(args.log_path, 'log_path', option_value=args.log_path)
    pysam_for_all_indel_bases = command_option_from(args.pysam_for_all_indel_bases, 'pysam_for_all_indel_bases')
    haploid_precision_mode = command_option_from(args.haploid_precision, 'haploid_precision')
    haploid_sensitive_mode = command_option_from(args.haploid_sensitive, 'haploid_sensitive')
    output_for_ensemble = command_option_from(args.output_for_ensemble, 'output_for_ensemble')
    debug = command_option_from(args.debug, 'debug')
    qual = command_option_from(args.qual, 'qual', option_value=args.qual)
    fast_plotting = command_option_from(args.fast_plotting, 'fast_plotting')

    call_var_bam_command_options = [
        ExecuteCommand('python', callVarBamBin),
        CommandOption('chkpnt_fn', chkpnt_fn),
        CommandOption('ref_fn', ref_fn),
        CommandOption('bam_fn', bam_fn),
        CommandOption('threshold', af_threshold),
        CommandOption('minCoverage', minCoverage),
        CommandOption('pypy', pypyBin),
        CommandOption('samtools', samtoolsBin),
        CommandOption('delay', delay),
        CommandOption('threads', threads),
        CommandOption('sampleName', sampleName),
        # optional command options
        CommandOption('vcf_fn', vcf_fn) if vcf_fn is not None else None,
        qual,
        stop_consider_left_edge,
        debug,
        pysam_for_all_indel_bases,
        haploid_precision_mode,
        haploid_sensitive_mode,
        output_for_ensemble,
    ]

    activation_only_command_options = [
        CommandOptionWithNoValue('activation_only'),
        log_path,
        CommandOption('max_plot', args.max_plot),
        CommandOption('parallel_level', args.parallel_level),
        CommandOption('workers', args.workers),
        fast_plotting,
    ] if args.activation_only else []

    is_bed_file_provided = bed_fn is not None
    command_string = command_string_from(call_var_bam_command_options + activation_only_command_options)

    with open(fai_fn, 'r') as fai_fp:
        for row in fai_fp:
            columns = row.strip().split("\t")

            contig_name = columns[0]
            if not is_include_all_contigs and str(contig_name) not in major_contigs:
                continue

            region_start, region_end = 0, 0
            contig_length = int(columns[1])
            while region_end < contig_length:
                region_start = region_end
                region_end = region_start + region_chunk_size
                if region_end > contig_length:
                    region_end = contig_length
                output_fn = "%s.%s_%d_%d.vcf" % (output_prefix, contig_name, region_start, region_end)

                is_region_in_bed = is_bed_file_provided and is_region_in(tree, contig_name, region_start, region_end)
                need_output_command = not is_bed_file_provided or is_region_in_bed
                if not need_output_command:
                    continue

                additional_command_options = [
                    CommandOption('ctgName', contig_name),
                    CommandOption('ctgStart', region_start),
                    CommandOption('ctgEnd', region_end),
                    CommandOption('call_fn', output_fn),
                    CommandOption('bed_fn', bed_fn) if is_region_in_bed else None
                ]
                print(command_string + " " + command_string_from(additional_command_options))


def main():
    parser = argparse.ArgumentParser(
        description="Create commands for calling variants in parallel using a trained model and a BAM file")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a model")

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

    parser.add_argument('--stop_consider_left_edge', action='store_true',
                        help="If not set, would consider left edge only. That is, count the left-most base-pairs of a read for coverage even if the starting position of a read is after the starting position of a tensor")

    parser.add_argument('--samtools', type=str, default="samtools",
                        help="Path to the 'samtools', default: %(default)s")

    parser.add_argument('--pypy', type=str, default="pypy3",
                        help="Path to the 'pypy', default: %(default)s")

    parser.add_argument('--delay', type=int, default=10,
                        help="Wait a short while for no more than %(default)s to start the job. This is to avoid starting multiple jobs simultaneously that might use up the maximum number of threads allowed, because Tensorflow will create more threads than needed at the beginning of running the program.")

    parser.add_argument('--debug', action='store_true',
                        help="Debug mode, optional")

    parser.add_argument('--pysam_for_all_indel_bases', action='store_true',
                        help="Always using pysam for outputting indel bases, optional")

    parser.add_argument('--haploid_precision', action='store_true',
                        help="call haploid instead of diploid (output homo-variant only)")
    parser.add_argument('--haploid_sensitive', action='store_true',
                        help="call haploid instead of diploid (output non-multi-variant only)")

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

    parser.add_argument('--output_for_ensemble', action='store_true',
                        help="Output for ensemble")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    if not args.includingAllContigs:
        print("echo \"[INFO] --includingAllContigs not enabled, use chr{1..22,X,Y,M,MT} and {1..22,X,Y,MT} by default\"\n")
    else:
        print("echo \"[INFO] --includingAllContigs enabled\"\n")

    Run(args)


if __name__ == "__main__":
    main()
