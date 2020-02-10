import sys
import shlex
import subprocess
import multiprocessing
import signal
import random
from os.path import dirname
from time import sleep
from argparse import ArgumentParser

from shared.command_options import (
    CommandOption,
    CommandOptionWithNoValue,
    ExecuteCommand,
    command_string_from,
    command_option_from
)
from shared.utils import file_path_from, executable_command_string_from


class InstancesClass(object):
    def __init__(self):
        self.extract_variant_candidate = None
        self.create_tensor = None
        self.call_variant = None

    def poll(self):
        self.extract_variant_candidate.poll()
        self.create_tensor.poll()
        self.call_variant.poll()


c = InstancesClass()


def check_return_code(signum, frame):
    c.poll()
    #print >> sys.stderr, c.extract_variant_candidate.returncode, c.create_tensor.returncode, c.call_variant.returncode
    if c.extract_variant_candidate.returncode != None and c.extract_variant_candidate.returncode != 0:
        c.create_tensor.kill()
        c.call_variant.kill()
        sys.exit("ExtractVariantCandidates.py or GetTruth.py exited with exceptions. Exiting...")

    if c.create_tensor.returncode != None and c.create_tensor.returncode != 0:
        c.extract_variant_candidate.kill()
        c.call_variant.kill()
        sys.exit("CreateTensor.py exited with exceptions. Exiting...")

    if c.call_variant.returncode != None and c.call_variant.returncode != 0:
        c.extract_variant_candidate.kill()
        c.create_tensor.kill()
        sys.exit("call_variant.py exited with exceptions. Exiting...")

    if (
        c.extract_variant_candidate.returncode == None or
        c.create_tensor.returncode == None or
        c.call_variant.returncode == None
    ):
        signal.alarm(5)


def Run(args):
    basedir = dirname(__file__)
    EVCBin = basedir + "/../clair.py ExtractVariantCandidates"
    GTBin = basedir + "/../clair.py GetTruth"
    CTBin = basedir + "/../clair.py CreateTensor"
    CVBin = basedir + "/../clair.py call_var"

    pypyBin = executable_command_string_from(args.pypy, exit_on_not_found=True)
    samtoolsBin = executable_command_string_from(args.samtools, exit_on_not_found=True)

    chkpnt_fn = file_path_from(args.chkpnt_fn, suffix=".meta", exit_on_not_found=True)
    bam_fn = file_path_from(args.bam_fn, exit_on_not_found=True)
    ref_fn = file_path_from(args.ref_fn, exit_on_not_found=True)
    vcf_fn = file_path_from(args.vcf_fn)
    bed_fn = file_path_from(args.bed_fn)

    dcov = args.dcov
    call_fn = args.call_fn
    af_threshold = args.threshold
    minCoverage = int(args.minCoverage)
    sampleName = args.sampleName
    ctgName = args.ctgName
    if ctgName is None:
        sys.exit("--ctgName must be specified. You can call variants on multiple chromosomes simultaneously.")

    stop_consider_left_edge = command_option_from(args.stop_consider_left_edge, 'stop_consider_left_edge')
    log_path = command_option_from(args.log_path, 'log_path', option_value=args.log_path)
    pysam_for_all_indel_bases = command_option_from(args.pysam_for_all_indel_bases, 'pysam_for_all_indel_bases')
    haploid_mode = command_option_from(args.haploid, 'haploid')
    output_for_ensemble = command_option_from(args.output_for_ensemble, 'output_for_ensemble')
    debug = command_option_from(args.debug, 'debug')
    qual = command_option_from(args.qual, 'qual', option_value=args.qual)
    fast_plotting = command_option_from(args.fast_plotting, 'fast_plotting')

    ctgStart = None
    ctgEnd = None
    if args.ctgStart is not None and args.ctgEnd is not None and int(args.ctgStart) <= int(args.ctgEnd):
        ctgStart = CommandOption('ctgStart', args.ctgStart)
        ctgEnd = CommandOption('ctgEnd', args.ctgEnd)

    if args.threads is None:
        numCpus = multiprocessing.cpu_count()
    else:
        numCpus = args.threads if args.threads < multiprocessing.cpu_count() else multiprocessing.cpu_count()

    maxCpus = multiprocessing.cpu_count()
    _cpuSet = ",".join(str(x) for x in random.sample(range(0, maxCpus), numCpus))

    taskSet = "taskset -c %s" % (_cpuSet)
    try:
        subprocess.check_output("which %s" % ("taskset"), shell=True)
    except:
        taskSet = ""

    if args.delay > 0:
        delay = random.randrange(0, args.delay)
        print("Delay %d seconds before starting variant calling ..." % (delay), file=sys.stderr)
        sleep(delay)

    extract_variant_candidate_command_options = [
        pypyBin,
        EVCBin,
        CommandOption('bam_fn', bam_fn),
        CommandOption('ref_fn', ref_fn),
        CommandOption('bed_fn', bed_fn),
        CommandOption('ctgName', ctgName),
        ctgStart,
        ctgEnd,
        CommandOption('threshold', af_threshold),
        CommandOption('minCoverage', minCoverage),
        CommandOption('samtools', samtoolsBin)
    ]
    get_truth_command_options = [
        pypyBin,
        GTBin,
        CommandOption('vcf_fn', vcf_fn),
        CommandOption('ctgName', ctgName),
        ctgStart,
        ctgEnd
    ]

    create_tensor_command_options = [
        pypyBin,
        CTBin,
        CommandOption('bam_fn', bam_fn),
        CommandOption('ref_fn', ref_fn),
        CommandOption('ctgName', ctgName),
        ctgStart,
        ctgEnd,
        stop_consider_left_edge,
        CommandOption('samtools', samtoolsBin),
        CommandOption('dcov', dcov)
    ]

    call_variant_command_options = [
        taskSet,
        ExecuteCommand('python', CVBin),
        CommandOption('chkpnt_fn', chkpnt_fn),
        CommandOption('call_fn', call_fn),
        CommandOption('bam_fn', bam_fn),
        CommandOption('sampleName', sampleName),
        CommandOption('threads', numCpus),
        CommandOption('ref_fn', ref_fn),
        pysam_for_all_indel_bases,
        haploid_mode,
        output_for_ensemble,
        qual,
        debug
    ]
    call_variant_with_activation_command_options = [
        CommandOptionWithNoValue('activation_only'),
        log_path,
        CommandOption('max_plot', args.max_plot),
        CommandOption('parallel_level', args.parallel_level),
        CommandOption('workers', args.workers),
        fast_plotting,
    ] if args.activation_only else []

    is_true_variant_call = vcf_fn is not None
    try:
        c.extract_variant_candidate = subprocess.Popen(
            shlex.split(command_string_from(
                get_truth_command_options if is_true_variant_call else extract_variant_candidate_command_options
            )),
            stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=8388608
        )

        c.create_tensor = subprocess.Popen(
            shlex.split(command_string_from(create_tensor_command_options)),
            stdin=c.extract_variant_candidate.stdout, stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=8388608
        )

        c.call_variant = subprocess.Popen(
            shlex.split(command_string_from(
                call_variant_command_options + call_variant_with_activation_command_options
            )),
            stdin=c.create_tensor.stdout, stdout=sys.stderr, stderr=sys.stderr, bufsize=8388608
        )
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit("Failed to start required processes. Exiting...")

    signal.signal(signal.SIGALRM, check_return_code)
    signal.alarm(2)

    try:
        c.call_variant.wait()
        c.create_tensor.stdout.close()
        c.create_tensor.wait()
        c.extract_variant_candidate.stdout.close()
        c.extract_variant_candidate.wait()
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt received when waiting at CallVarBam, terminating all scripts.")
        try:
            c.call_variant.terminate()
            c.create_tensor.terminate()
            c.extract_variant_candidate.terminate()
        except Exception as e:
            print(e)

        raise KeyboardInterrupt
    except Exception as e:
        print("Exception received when waiting at CallVarBam, terminating all scripts.")
        print(e)
        try:
            c.call_variant.terminate()
            c.create_tensor.terminate()
            c.extract_variant_candidate.terminate()
        except Exception as e:
            print(e)

        raise e


def main():
    parser = ArgumentParser(description="Call variants using a trained model and a BAM file")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a model")

    parser.add_argument('--ref_fn', type=str, default="ref.fa",
                        help="Reference fasta file input, default: %(default)s")

    parser.add_argument('--bed_fn', type=str, default=None,
                        help="Call variant only in these regions, works in intersection with ctgName, ctgStart and ctgEnd, optional, default: as defined by ctgName, ctgStart and ctgEnd")

    parser.add_argument('--bam_fn', type=str, default="bam.bam",
                        help="BAM file input, default: %(default)s")

    parser.add_argument('--call_fn', type=str, default=None,
                        help="Output variant predictions")

    parser.add_argument('--vcf_fn', type=str, default=None,
                        help="Candidate sites VCF file input, if provided, variants will only be called at the sites in the VCF file,  default: %(default)s")

    parser.add_argument('--threshold', type=float, default=0.125,
                        help="Minimum allele frequence of the 1st non-reference allele for a site to be considered as a condidate site, default: %(default)f")

    parser.add_argument('--minCoverage', type=float, default=4,
                        help="Minimum coverage required to call a variant, default: %(default)d")

    parser.add_argument('--qual', type=int, default=None,
                        help="If set, variant with equal or higher quality will be marked PASS, or LowQual otherwise, optional")

    parser.add_argument('--sampleName', type=str, default="SAMPLE",
                        help="Define the sample name to be shown in the VCF file")

    parser.add_argument('--ctgName', type=str, default=None,
                        help="The name of sequence to be processed, default: %(default)s")
    parser.add_argument('--ctgStart', type=int, default=None,
                        help="The 1-based starting position of the sequence to be processed")
    parser.add_argument('--ctgEnd', type=int, default=None,
                        help="The 1-based inclusive ending position of the sequence to be processed")

    parser.add_argument('--stop_consider_left_edge', action='store_true',
                        help="If not set, would consider left edge only. That is, count the left-most base-pairs of a read for coverage even if the starting position of a read is after the starting position of a tensor")

    parser.add_argument('--dcov', type=int, default=250,
                        help="Cap depth per position at %(default)s")

    parser.add_argument('--samtools', type=str, default="samtools",
                        help="Path to the 'samtools', default: %(default)s")

    parser.add_argument('--pypy', type=str, default="pypy3",
                        help="Path to the 'pypy', default: %(default)s")

    parser.add_argument('--threads', type=int, default=None,
                        help="Number of threads, optional")

    parser.add_argument('--delay', type=int, default=10,
                        help="Wait a short while for no more than %(default)s to start the job. This is to avoid starting multiple jobs simultaneously that might use up the maximum number of threads allowed, because Tensorflow will create more threads than needed at the beginning of running the program.")

    parser.add_argument('--debug', action='store_true',
                        help="Debug mode, optional")

    parser.add_argument('--pysam_for_all_indel_bases', action='store_true',
                        help="Always using pysam for outputting indel bases, optional")

    parser.add_argument('--haploid', action='store_true',
                        help="call haploid instead of diploid")

    parser.add_argument('--activation_only', action='store_true',
                        help="Output activation only, no prediction")
    parser.add_argument('--max_plot', type=int, default=10,
                        help="The maximum number of plots output, negative number means no limit (plot all), default: %(default)s")
    parser.add_argument('--log_path', type=str, nargs='?', default=None,
                        help="The path for tensorflow logging, default: %(default)s")
    parser.add_argument('-p', '--parallel_level', type=int, default=2,
                        help="The level of parallelism in plotting (currently available: 0, 2), default: %(default)s")
    parser.add_argument('--fast_plotting', action='store_true',
                        help="Enable fast plotting.")
    parser.add_argument('-w', '--workers', type=int, default=8,
                        help="The number of workers in plotting, default: %(default)s")

    parser.add_argument('--output_for_ensemble', action='store_true',
                        help="Output for ensemble")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)


if __name__ == "__main__":
    main()
