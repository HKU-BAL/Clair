import sys
import os
import time
import argparse
import param
import logging
import numpy as np
from threading import Thread
from math import log, e
from enum import Enum

import utils
import clair as cv

logging.basicConfig(format='%(message)s', level=logging.INFO)
num2base = dict(zip((0, 1, 2, 3), "ACGT"))
base2num = dict(zip("ACGT", (0, 1, 2, 3)))
v1Type2Name = dict(zip((0, 1, 2, 3, 4), ('HET', 'HOM', 'INS', 'DEL', 'REF')))
v2Zygosity2Name = dict(zip((0, 1), ('HET', 'HOM')))
v2Type2Name = dict(zip((0, 1, 2, 3), ('REF', 'SNP', 'INS', 'DEL')))
v2Length2Name = dict(zip((0, 1, 2, 3, 4, 5), ('0', '1', '2', '3', '4', '4+')))
maximum_variant_length = 5
inferred_indel_length_minimum_allele_frequency = 0.125


class CountIndex(Enum):
    reference = 0
    insert = 1
    delete = 2
    snp = 3


class GenoTypeIndex(Enum):
    homo_reference = 0  # 0/0
    homo_variant = 1    # 1/1
    hetero_variant = 2  # 0/1 OR 1/2

def geno_type_string_from(geno_type_index):
    if geno_type_index == GenoTypeIndex.homo_reference:
        return "0/0"
    elif geno_type_index == GenoTypeIndex.homo_variant:
        return "1/1"
    elif geno_type_index == GenoTypeIndex.hetero_variant:
        return "0/1"
    return ""


class BaseChangeIndex(Enum):
    AA = 0
    AC = 1
    AG = 2
    AT = 3
    CC = 4
    CG = 5
    CT = 6
    GG = 7
    GT = 8
    TT = 9
    DelDel = 10
    ADel = 11
    CDel = 12
    GDel = 13
    TDel = 14
    InsIns = 15
    AIns = 16
    CIns = 17
    GIns = 18
    TIns = 19
    InsDel = 20


def is_reference_from(base_change_index, geno_type_index):
    return (
        geno_type_index == GenoTypeIndex.homo_reference and
        (
            base_change_index == BaseChangeIndex.AA or
            base_change_index == BaseChangeIndex.CC or
            base_change_index == BaseChangeIndex.GG or
            base_change_index == BaseChangeIndex.TT
        )
    )


def is_SNP_from(base_change_index, geno_type_index):
    return (
        (
            geno_type_index == GenoTypeIndex.homo_variant and
            (
                base_change_index == BaseChangeIndex.AA or
                base_change_index == BaseChangeIndex.CC or
                base_change_index == BaseChangeIndex.GG or
                base_change_index == BaseChangeIndex.TT
            )
        ) or
        (
            geno_type_index == GenoTypeIndex.hetero_variant and
            (
                base_change_index == BaseChangeIndex.AC or
                base_change_index == BaseChangeIndex.AG or
                base_change_index == BaseChangeIndex.AT or
                base_change_index == BaseChangeIndex.CG or
                base_change_index == BaseChangeIndex.CT or
                base_change_index == BaseChangeIndex.GT
            )
        )
    )


def is_indel_insertion_from(base_change_index, geno_type_index):
    return (
        (
            geno_type_index == GenoTypeIndex.homo_variant and
            base_change_index == BaseChangeIndex.InsIns
        ) or
        (
            geno_type_index == GenoTypeIndex.hetero_variant and
            (
                base_change_index == BaseChangeIndex.InsIns or
                base_change_index == BaseChangeIndex.AIns or
                base_change_index == BaseChangeIndex.CIns or
                base_change_index == BaseChangeIndex.GIns or
                base_change_index == BaseChangeIndex.TIns
            )
        )
    )


def is_indel_deletion_from(base_change_index, geno_type_index):
    return (
        (
            geno_type_index == GenoTypeIndex.homo_variant and
            base_change_index == BaseChangeIndex.DelDel
        ) or
        (
            geno_type_index == GenoTypeIndex.hetero_variant and
            (
                base_change_index == BaseChangeIndex.DelDel or
                base_change_index == BaseChangeIndex.ADel or
                base_change_index == BaseChangeIndex.CDel or
                base_change_index == BaseChangeIndex.GDel or
                base_change_index == BaseChangeIndex.TDel
            )
        )
    )


def base_change_label_from(base_change_index):
    return [
        'AA',
        'AC',
        'AG',
        'AT',
        'CC',
        'CG',
        'CT',
        'GG',
        'GT',
        'TT',
        'DelDel',
        'ADel',
        'CDel',
        'GDel',
        'TDel',
        'InsIns',
        'AIns',
        'CIns',
        'GIns',
        'TIns',
        'InsDel'
    ][base_change_index]

def reference_or_snp_bases_from(base_change_index):
    try :
        base_change_label = base_change_label_from(base_change_index)
        return base_change_label[0], base_change_label[1]
    except:
        return "", ""



def Run(args):
    utils.setup_environment()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    if args.threads == None:
        if args.tensor_fn == "PIPE":
            param.NUM_THREADS = 4
    else:
        param.NUM_THREADS = args.threads
        param.NUM_THREADS -= 1
        if param.NUM_THREADS < 1:
            param.NUM_THREADS = 1

    m = cv.Clair()
    m.init()
    m.restore_parameters(os.path.abspath(args.chkpnt_fn))

    if args.activation_only:
        log_activation(args, m, utils)
    else:
        Test(args, m, utils)


def Output(
    args,
    call_fh,
    batch_size,
    X,
    posBatch,
    base_change_probabilities,
    geno_type_probabilities,
    variant_length_float
):
    if len(base_change_probabilities) != batch_size:
        sys.exit(
            "Inconsistent shape between input tensor and output predictions %d/%d" %
            (batch_size, len(base_change_probabilities))
        )

    is_show_reference = args.showRef
    flanking_base_number = param.flankingBaseNum
    position_center = flanking_base_number
    no_of_rows = len(base_change_probabilities)

    for row_index in range(no_of_rows):
        base_change_index = np.argmax(base_change_probabilities[row_index])
        geno_type_index = np.argmax(geno_type_probabilities[row_index])

        is_reference = is_reference_from(base_change_index, geno_type_index)
        is_SNP = is_SNP_from(base_change_index, geno_type_index)
        is_indel_insertion = is_indel_insertion_from(base_change_index, geno_type_index)
        is_indel_deletion = is_indel_deletion_from(base_change_index, geno_type_index)

        # show reference / not show reference handling
        if not is_show_reference and is_reference:
            continue

        # Get Indel Length
        variant_length = int(variant_length_float)

        # get chromosome, position and
        # reference bases with flanking "flanking_base_number" flanking bases at position
        chromosome, position, reference_sequence = posBatch[row_index].split(":")
        position = int(position)

        # Get genotype quality
        sorted_base_change_probabilities = np.sort(base_change_probabilities)[::-1]
        sorted_geno_type_probabilities = np.sort(geno_type_probabilities)[::-1]
        quality_score = int(
            (-10 * log(e, 10)) * log(
                (
                    sorted_base_change_probabilities[1] ** 1.0 *
                    sorted_geno_type_probabilities[1] ** 1.0 + 1e-300
                ) /
                (
                    sorted_base_change_probabilities[0] *
                    sorted_geno_type_probabilities[0] + 1e-300
                )
            )
        )
        if quality_score > 999:
            quality_score = 999

        # filtration value
        filtration_value = "."
        if args.qual != None:
            if quality_score >= args.qual:
                filtration_value = "PASS"
            else:
                filtration_value = "LowQual"

        # Initialize other variables
        reference_base = ""
        alternate_base = ""
        inferred_indel_length = 0
        read_depth = 0
        allele_frequency = 0.
        info = []

        if is_SNP or is_reference:
            read_depth = sum(
                X[row_index, position_center, :, CountIndex.reference] +
                X[row_index, position_center, :, CountIndex.snp]
            )
        elif is_indel_insertion:
            read_depth = sum(
                X[row_index, position_center+1, :, CountIndex.reference] +
                X[row_index, position_center+1, :, CountIndex.insert]
            )
        elif is_indel_deletion:
            read_depth = sum(
                X[row_index, position_center+1, :, CountIndex.reference] +
                X[row_index, position_center+1, :, CountIndex.delete]
            )
        else:
            # TODO:
            # handle collision cases:
            # - is homo-reference genotype but base_change doesn't reflect that or vice versa
            # - is homo-variant (SNP) or hetero-variant (SNP) but base_change doesn't reflect that or vice versa
            # - insertion only but base_change doesn't reflect that or vice versa
            # - deletion only but base_change doesn't reflect that or vice versa
            # - mixture of insertion and deletion
            pass

        if read_depth == 0:
            continue

        if is_SNP or is_reference:
            base1, base2 = reference_or_snp_bases_from(base_change_index)
            reference_base = reference_sequence[flanking_base_number]
            alternate_base = reference_base if is_reference else (base1 if base1 != reference_base else base2)

            if read_depth != 0:
                allele_frequency = (
                    X[row_index, position_center, base2num[alternate_base], CountIndex.snp] +
                    X[row_index, position_center, base2num[alternate_base]+4, CountIndex.snp]
                ) / read_depth

        elif is_indel_insertion:
            if variant_length == 0:
                continue

            if read_depth != 0:
                allele_frequency = sum(X[row_index, position_center+1, :, CountIndex.insert]) / read_depth

            if variant_length != maximum_variant_length:
                for k in range(flanking_base_number + 1, flanking_base_number + variant_length + 1):
                    alternate_base += num2base[np.argmax(X[row_index, k, :, CountIndex.insert]) % 4]
            else:
                for k in range(flanking_base_number + 1, 2*flanking_base_number + 1):
                    referenceTensor = X[row_index, k, :, CountIndex.reference]
                    insertionTensor = X[row_index, k, :, CountIndex.insert]
                    if (
                        k < (flanking_base_number + maximum_variant_length) or
                        sum(insertionTensor) >= (inferred_indel_length_minimum_allele_frequency * sum(referenceTensor))
                    ):
                        inferred_indel_length += 1
                        alternate_base += num2base[np.argmax(insertionTensor) % 4]
                    else:
                        break
            reference_base = reference_sequence[position_center]

            # insertions longer than (flanking_base_number-1) are marked SV
            if inferred_indel_length >= flanking_base_number:
                alternate_base = "<INS>"
                info.append("SVTYPE=INS")
            else:
                alternate_base = reference_base + alternate_base

        elif is_indel_deletion:
            if variant_length == 0:
                continue

            if read_depth != 0:
                allele_frequency = sum(X[row_index, position_center+1, :, CountIndex.delete]) / read_depth

            # infer the deletion length
            if variant_length == maximum_variant_length:
                for k in range(flanking_base_number+1, 2*flanking_base_number + 1):
                    if (
                        k < (flanking_base_number + maximum_variant_length) or
                        sum(X[row_index, k, :, CountIndex.delete]) >= (
                            inferred_indel_length_minimum_allele_frequency * sum(X[row_index, k, :, CountIndex.reference]))
                    ):
                        inferred_indel_length += 1
                    else:
                        break

            # deletions longer than (flanking_base_number-1) are marked SV
            if inferred_indel_length >= flanking_base_number:
                reference_base = reference_sequence[flanking_base_number]
                alternate_base = "<DEL>"
                info.append("SVTYPE=DEL")
            elif variant_length != maximum_variant_length:
                reference_base = reference_sequence[flanking_base_number:flanking_base_number+variant_length + 1]
                alternate_base = reference_sequence[flanking_base_number]
            else:
                reference_base = reference_sequence[flanking_base_number:flanking_base_number+inferred_indel_length + 1]
                alternate_base = reference_sequence[flanking_base_number]

        else:
            # TODO:
            # handle collision cases:
            # - is homo-reference genotype but base_change doesn't reflect that or vice versa
            # - is homo-variant (SNP) or hetero-variant (SNP) but base_change doesn't reflect that or vice versa
            # - insertion only but base_change doesn't reflect that or vice versa
            # - deletion only but base_change doesn't reflect that or vice versa
            # - mixture of insertion and deletion
            pass

        if inferred_indel_length > 0 and inferred_indel_length < flanking_base_number:
            info.append("LENGUESS=%d" % inferred_indel_length)

        information_string = ""
        if len(info) == 0:
            information_string = "."
        else:
            information_string = ";".join(info)

        genotype_string = geno_type_string_from(geno_type_index)

        print >> call_fh, "%s\t%d\t.\t%s\t%s\t%d\t%s\t%s\tGT:GQ:DP:AF\t%s:%d:%d:%.4f" % (
            chromosome,
            position,
            reference_base,
            alternate_base,
            quality_score,
            filtration_value,
            information_string,
            genotype_string,
            quality_score,
            read_depth,
            allele_frequency if read_depth != 0 else 0.0
        )


def print_vcf_header(args, call_fh):
    print >> call_fh, '##fileformat=VCFv4.1'
    print >> call_fh, '##FILTER=<ID=PASS,Description="All filters passed">'
    print >> call_fh, '##FILTER=<ID=LowQual,Description="Confidence in this variant being real is below calling threshold.">'
    print >> call_fh, '##ALT=<ID=DEL,Description="Deletion">'
    print >> call_fh, '##ALT=<ID=INS,Description="Insertion of novel sequence">'
    print >> call_fh, '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">'
    print >> call_fh, '##INFO=<ID=LENGUESS,Number=.,Type=Integer,Description="Best guess of the indel length">'
    print >> call_fh, '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">'
    print >> call_fh, '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">'
    print >> call_fh, '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">'
    print >> call_fh, '##FORMAT=<ID=AF,Number=1,Type=Float,Description="Estimated allele frequency in the range (0,1)">'

    if args.ref_fn != None:
        fai_fn = args.ref_fn + ".fai"
        fai_fp = open(fai_fn)
        for line in fai_fp:
            fields = line.strip().split("\t")
            chromName = fields[0]
            chromLength = int(fields[1])
            print >> call_fh, "##contig=<ID=%s,length=%d>" % (chromName, chromLength)

    print >> call_fh, '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t%s' % (args.sampleName)


def log_activation(args, m, utils):
    if args.log_path is None:
        return

    summary_writer = m.get_summary_file_writer(args.log_path)

    if summary_writer is None:
        return

    tensorGenerator = utils.GetTensor(args.tensor_fn, param.predictBatchSize)
    logging.info("Plotting activations ...")

    num_plotted = 0
    while(num_plotted < args.max_plot or args.max_plot < 0):
        print("Getting next batch")
        is_end_of_generator, batch_size, batch_X, batch_positions = next(tensorGenerator)
        print("Batch generation complete %d" % batch_size)
        # strip away the reference string, keeping the chr and coor only
        batch_positions = [s[:s.rfind(":")] for s in batch_positions]
        summaries = m.get_activation_summary(batch_X, operations=m.layers, batch_item_suffixes=batch_positions,
                                             max_plot_in_batch=args.max_plot - num_plotted if args.max_plot >= 0 else batch_size, parallel_level=args.parallel_level, num_workers=args.workers, fast_plotting=args.fast_plotting)
        for summary in summaries:
            summary_writer.add_summary(summary)
        num_plotted += min(batch_size, args.max_plot - num_plotted if args.max_plot >= 0 else batch_size)
        if is_end_of_generator == 1:
            break
    print("Finished plotting %d" % num_plotted)


def Test(args, m, utils):
    call_fh = open(args.call_fn, "w")

    print_vcf_header(args, call_fh)

    tensorGenerator = utils.GetTensor(args.tensor_fn, param.predictBatchSize)
    logging.info("Calling variants ...")
    predictStart = time.time()
    end = 0
    end2 = 0
    terminate = 0
    end2, num2, XBatch2, posBatch2 = next(tensorGenerator)
    m.predict(XBatch2, result_caching=True)
    base = m.predictBaseRTVal
    gt = m.predictGenoTypeRTVal
    l = m.predictIndelLengthRTVal
    if end2 == 0:
        end = end2
        num = num2
        XBatch = XBatch2
        posBatch = posBatch2
        end2, num2, XBatch2, posBatch2 = next(tensorGenerator)
        while True:
            if end == 1:
                terminate = 1
            threadPool = []
            if end == 0:
                threadPool.append(Thread(target=m.predict, args=(XBatch2, True)))
            threadPool.append(Thread(target=Output, args=(args, call_fh, num, XBatch, posBatch, base, gt, l, )))
            for t in threadPool:
                t.start()
            if end2 == 0:
                end3, num3, XBatch3, posBatch3 = next(tensorGenerator)
            for t in threadPool:
                t.join()
            base = m.predictBaseRTVal
            gt = m.predictGenoTypeRTVal
            l = m.predictIndelLengthRTVal
            if end == 0:
                end = end2
                num = num2
                XBatch = XBatch2
                posBatch = posBatch2
            if end2 == 0:
                end2 = end3
                num2 = num3
                XBatch2 = XBatch3
                posBatch2 = posBatch3
            #print >> sys.stderr, end, end2, end3, terminate
            if terminate == 1:
                break
    elif end2 == 1:
        Output(args, call_fh, num2, XBatch2, posBatch2, base, gt, l)

    logging.info("Total time elapsed: %.2f s" % (time.time() - predictStart))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Call variants using a trained Clair model and tensors of candididate variants")

    parser.add_argument('--tensor_fn', type=str, default="PIPE",
                        help="Tensor input, use PIPE for standard input")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a checkpoint for testing or continue training")

    parser.add_argument('--call_fn', type=str, default=None,
                        help="Output variant predictions")

    parser.add_argument('--qual', type=int, default=None,
                        help="If set, variant with equal or higher quality will be marked PASS, or LowQual otherwise, optional")

    parser.add_argument('--sampleName', type=str, default="SAMPLE",
                        help="Define the sample name to be shown in the VCF file")

    parser.add_argument('--showRef', type=param.str2bool, nargs='?', const=True, default=False,
                        help="Show reference calls, optional")

    parser.add_argument('--ref_fn', type=str, default=None,
                        help="Reference fasta file input, optional, print contig tags in the VCF header if set")

    parser.add_argument('--threads', type=int, default=None,
                        help="Number of threads, optional")

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

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)
