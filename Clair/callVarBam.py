import os
import sys
import argparse
import param
import shlex
import subprocess
import multiprocessing
import signal
import random
import time
from collections import namedtuple

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


class InstancesClass(object):
    def __init__(self):
        self.EVCInstance = None
        self.CTInstance = None
        self.CVInstance = None

    def poll(self):
        self.EVCInstance.poll()
        self.CTInstance.poll()
        self.CVInstance.poll()


c = InstancesClass()


def CheckRtCode(signum, frame):
    c.poll()
    #print >> sys.stderr, c.EVCInstance.returncode, c.CTInstance.returncode, c.CVInstance.returncode
    if c.EVCInstance.returncode != None and c.EVCInstance.returncode != 0:
        c.CTInstance.kill()
        c.CVInstance.kill()
        sys.exit("ExtractVariantCandidates.py or GetTruth.py exited with exceptions. Exiting...")

    if c.CTInstance.returncode != None and c.CTInstance.returncode != 0:
        c.EVCInstance.kill()
        c.CVInstance.kill()
        sys.exit("CreateTensors.py exited with exceptions. Exiting...")

    if c.CVInstance.returncode != None and c.CVInstance.returncode != 0:
        c.EVCInstance.kill()
        c.CTInstance.kill()
        sys.exit("call_var.py exited with exceptions. Exiting...")

    if c.EVCInstance.returncode == None or c.CTInstance.returncode == None or c.CVInstance.returncode == None:
        signal.alarm(5)


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
    EVCBin = CheckFileExist(basedir + "/../dataPrepScripts/ExtractVariantCandidates.py")
    GTBin = CheckFileExist(basedir + "/../dataPrepScripts/GetTruth.py")
    CTBin = CheckFileExist(basedir + "/../dataPrepScripts/CreateTensor.py")
    CVBin = CheckFileExist(basedir + "/call_var.py")

    pypyBin = CheckCmdExist(args.pypy)
    samtoolsBin = CheckCmdExist(args.samtools)

    chkpnt_fn = CheckFileExist(args.chkpnt_fn, sfx=".meta")
    bam_fn = CheckFileExist(args.bam_fn)
    ref_fn = CheckFileExist(args.ref_fn)
    vcf_fn = CheckFileExist(args.vcf_fn) if args.vcf_fn != None else None
    bed_fn = CheckFileExist(args.bed_fn) if args.bed_fn != None else ""

    dcov = args.dcov
    call_fn = args.call_fn
    threshold = args.threshold
    minCoverage = args.minCoverage
    sampleName = args.sampleName
    qual = args.qual
    log_path= args.log_path
    fast_plotting= args.fast_plotting
    ctgName = args.ctgName
    ctgStart=args.ctgStart
    ctgEnd=args.ctgEnd

    if ctgName == None:
        sys.exit("--ctgName must be specified. You can call variants on multiple chromosomes simultaneously.")

    if args.considerleftedge:
        considerleftedge = CommandOptionWithNoValue('considerleftedge')
    else:
        considerleftedge=""

    if args.pysam_for_all_indel_bases:
        pysam_for_all_indel_bases = CommandOptionWithNoValue('pysam_for_all_indel_bases')
    else:
        pysam_for_all_indel_bases = ""

    if args.debug:
        debug = CommandOptionWithNoValue('debug')
    else:
        debug = ""

    if args.ctgStart != None and args.ctgEnd != None and int(args.ctgStart) <= int(args.ctgEnd):
        ctgStart = CommandOption('ctgStart',ctgStart)
        ctgEnd = CommandOption('ctgEnd',ctgEnd)
    else:
        ctgStart = ""
        ctgEnd = ""

    if args.threads == None:
        numCpus = multiprocessing.cpu_count()
    else:
        numCpus = args.threads if args.threads < multiprocessing.cpu_count() else multiprocessing.cpu_count()

    _cpuSet = ",".join(str(x) for x in random.sample(xrange(0, maxCpus), numCpus))
    taskSet = "taskset -c %s" % (_cpuSet)

    try:
        subprocess.check_output("which %s" % ("taskset"), shell=True)
    except:
        taskSet = ""

    if args.delay > 0:
        delay = random.randrange(0, args.delay)
        print >> sys.stderr, "Delay %d seconds before starting variant calling ..." % (delay)
        time.sleep(delay)

    vcfIsNone_commands=[
        pypyBin,
        EVCBin,
        CommandOption('bam_fn',bam_fn),
        CommandOption('ref_fn',ref_fn),
        CommandOption('bed_fn',bed_fn),
        CommandOption('ctgName',ctgName),
        ctgStart,
        ctgEnd,
        CommandOption('threshold',threshold),
        CommandOption('midCoverage',minCoverage),
        CommandOption('samtoolsBin',samtoolsBin)
    ]

    vcfIsNotNone_commands=[
        pypyBin,
        GTBin,
        CommandOption('vcf_fn',vcf_fn),
        CommandOption('ctgName',ctgName),
        ctgStart,
        ctgEnd
    ]

    required_commands=[
        pypyBin,
        CTBin,
        CommandOption('bam_fn',bam_fn),
        CommandOption('ref_fn',ref_fn),
        CommandOption('ctgName',ctgName),
        ctgStart,
        ctgEnd,
        considerleftedge,
        CommandOption('samtools',samtoolsBin),
        CommandOption('dcov',dcov),
        pysam_for_all_indel_bases
    ]

    activation_commands=[
        taskSet,
        ExecuteCommand('python',CVBin),
        CommandOption('chkpnt_fn', chkpnt_fn),
        CommandOption('call_fn',call_fn),
        CommandOption('bam_fn',bam_fn),
        CommandOption('sampleName',sampleName),
        CommandOption('threads',numCpus)
    ]

    activationOnly_commands=[
        CommandOptionWithNoValue('activation_only'),
        CommandOption('log_path',log_path),
        CommandOption('max_plot',args.max_plot),
        CommandOption('parallel_level',args.parallel_path),
        CommandOption('worker',args.workers),
        CommandOption('ref_fn',ref_fn),
        qual,
        fast_plotting,
        debug
    ]

    notActivationOnly_commands=[
        CommandOption('ref_fn',ref_fn),
        qual,
        debug
    ]


    try:
        if vcf_fn == None:
            c.EVCInstance = subprocess.Popen(
                shlex.split(executable_command_string_from(vcfIsNone_commands)),
                stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=8388608)
        else:
            c.EVCInstance = subprocess.Popen(
                shlex.split(executable_command_string_from(vcfIsNotNone_commands)),
                stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=8388608)
        c.CTInstance = subprocess.Popen(
            shlex.splie(executable_command_string_from(required_commands)),
            stdin=c.EVCInstance.stdout, stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=8388608)

        if args.activation_only:
            c.CVInstance = subprocess.Popen(
                shlex.split(executable_command_string_from(activation_commands+activationOnly_commands)),
                stdin=c.CTInstance.stdout, stdout=sys.stderr, stderr=sys.stderr, bufsize=8388608)
        else:
            c.CVInstance = subprocess.Popen(
                shlex.split(executable_command_string_from(activation_commands+notActivationOnly_commands)),
                stdin=c.CTInstance.stdout, stdout=sys.stderr, stderr=sys.stderr, bufsize=8388608)
    except Exception as e:
        print >> sys.stderr, e
        sys.exit("Failed to start required processes. Exiting...")

    signal.signal(signal.SIGALRM, CheckRtCode)
    signal.alarm(2)

    try:
        c.CVInstance.wait()
        c.CTInstance.stdout.close()
        c.CTInstance.wait()
        c.EVCInstance.stdout.close()
        c.EVCInstance.wait()
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt received when waiting at CallVarBam, terminating all scripts.")
        try:
            c.CVInstance.terminate()
        except Exception as e:
            print(e.message)
        try:
            c.CTInstance.terminate()
        except Exception as e:
            print(e.message)
        try:
            c.EVCInstance.terminate()
        except Exception as e:
            print(e.message)

        raise KeyboardInterrupt
    except Exception as e:
        print("Exception received when waiting at CallVarBam, terminating all scripts.")
        print(e.message)
        try:
            c.CVInstance.terminate()
        except Exception as e:
            print(e.message)
        try:
            c.CTInstance.terminate()
        except Exception as e:
            print(e.message)
        try:
            c.EVCInstance.terminate()
        except Exception as e:
            print("C")
            print(e.message)
        raise e


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Call variants using a trained Clair model and a BAM file")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a Clair model")

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

    parser.add_argument('--considerleftedge', type=param.str2bool, nargs='?', const=True, default=True,
                        help="Count the left-most base-pairs of a read for coverage even if the starting position of a read is after the starting position of a tensor, default: %(default)s")

    parser.add_argument('--dcov', type=int, default=250,
                        help="Cap depth per position at %(default)s")

    parser.add_argument('--samtools', type=str, default="samtools",
                        help="Path to the 'samtools', default: %(default)s")

    parser.add_argument('--pypy', type=str, default="pypy",
                        help="Path to the 'pypy', default: %(default)s")

    parser.add_argument('--threads', type=int, default=None,
                        help="Number of threads, optional")

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

    parser.add_argument('--fast_plotting', action='store_true',
                        help="Enable fast plotting.")

    parser.add_argument('-w', '--workers', type=int, default=8,
                        help="The number of workers in plotting, default: %(default)s")

    parser.add_argument('--debug', type=param.str2bool, nargs='?', const=True, default=False,
                        help="Debug mode, optional")

    parser.add_argument('--pysam_for_all_indel_bases', type=param.str2bool, nargs='?', const=True, default=False,
                        help="Always using pysam for outputting indel bases, optional")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)
