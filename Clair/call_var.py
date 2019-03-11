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
    insertion = 1
    deletion = 2
    snp = 3


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
    num,
    X,
    posBatch,
    base_change_probabilities,
    zygosity_probabilities,
    variant_type_probabilities,
    indel_length_probabilities
):
    if num != len(base_change_probabilities):
        sys.exit(
            "Inconsistent shape between input tensor and output predictions %d/%d" %
            (num, len(base_change_probabilities))
        )

    is_show_reference = args.showRef
    flanking_base_number = param.flankingBaseNum
    position_center = flanking_base_number
    no_of_rows = len(base_change_probabilities)
    #          --------------  ------  ------------    ------------------
    #          Base chng       Zygo.   Var type        Var length
    #          A   C   G   T   HET HOM REF SNP INS DEL 0   1   2   3   4   >=4
    #          0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
    for row_index in range(no_of_rows):
        # Get variant type, 0:REF, 1:SNP, 2:INS, 3:DEL
        variant_type = np.argmax(variant_type_probabilities[row_index])
        is_reference = variant_type == 0
        is_SNP = variant_type == 1
        is_indel_insertion = variant_type == 2
        is_indel_deletion = variant_type == 3

        # show reference / not show reference handling
        if not is_show_reference and is_reference:
            continue

        # Get zygosity, 0:HET, 1:HOM
        variant_zygosity = np.argmax(zygosity_probabilities[row_index])
        is_heterozygous = variant_zygosity == 0
        is_homozygous = variant_zygosity == 1

        # Get Indel Length, 0:0, 1:1, 2:2, 3:3, 4:4, 5:>4
        variant_length = np.argmax(indel_length_probabilities[row_index])
        # if variant_length  == 0:
        #     variant_type = 0
        #     continue
        # else:
        #     variant_type = 2

        # get chromosome, position and
        # reference bases with flanking "flanking_base_number" flanking bases at position
        chromosome, position, reference_sequence = posBatch[row_index].split(":")
        position = int(position)

        # Get genotype quality
        sorted_variant_type_probabilities = np.sort(variant_type_probabilities[row_index])[::-1]
        sorted_zygosity_probabilities = np.sort(zygosity_probabilities[row_index])[::-1]
        sorted_indel_length_probabilities = np.sort(indel_length_probabilities[row_index])[::-1]
        quality_score = int(
            (-10 * log(e, 10)) * log(
                (
                    sorted_variant_type_probabilities[1] ** 1.0 *
                    sorted_zygosity_probabilities[1] ** 1.0 *
                    sorted_indel_length_probabilities[1] ** 1.0 + 1e-300
                ) /
                (
                    sorted_variant_type_probabilities[0] *
                    sorted_zygosity_probabilities[0] *
                    sorted_indel_length_probabilities[0] + 1e-300
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

        # Get possible alternative bases
        sorted_base_change_probabilities = base_change_probabilities[row_index].argsort()[::-1]
        base1 = num2base[sorted_base_change_probabilities[0]]
        base2 = num2base[sorted_base_change_probabilities[1]]

        # Initialize other variables
        reference_base = ""
        alternate_base = ""
        inferred_indel_length = 0
        read_depth = 0
        allele_frequency = 0.
        info = []

        # read_depth = (
        #     sum(X[row_index, flanking_base_number, :, 0]) +
        #     sum(X[row_index, flanking_base_number, :, 1]) +
        #     sum(X[row_index, flanking_base_number, :, 2]) +
        #     sum(X[row_index, flanking_base_number, :, 3])
        # )
        # if read_depth == 0:
        #     continue

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
        if read_depth == 0:
            continue

        if is_SNP or is_reference:
            reference_base = reference_sequence[flanking_base_number]
            if is_SNP:
                alternate_base = base1 if base1 != reference_base else base2
            elif is_reference:
                alternate_base = reference_base
            # read_depth = sum(
            #     X[row_index, flanking_base_number, :, 0] + X[row_index, flanking_base_number, :, 3]
            # )
            if read_depth != 0:
                allele_frequency = (
                    X[row_index, position_center, base2num[alternate_base], CountIndex.snp] +
                    X[row_index, position_center, base2num[alternate_base]+4, CountIndex.snp]
                ) / read_depth

        elif is_indel_insertion:
            # infer the insertion length
            if variant_length == 0:
                variant_length = 1

            # read_depth = sum(
            #     X[row_index, flanking_base_number+1, :, 0] + X[row_index, flanking_base_number+1, :, 1]
            # )
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

        elif is_indel_deletion:  # DEL
            if variant_length == 0:
                variant_length = 1

            # read_depth = sum(
            #     X[row_index, flanking_base_number+1, :, 0] + X[row_index, flanking_base_number+1, :, 2]
            # )
            if read_depth != 0:
                allele_frequency = sum(X[row_index, position_center+1, :, CountIndex.delete]) / read_depth

            # infer the deletion length
            if variant_length == maximum_variant_length:
                for k in range(flanking_base_number+1, 2*flanking_base_number + 1):
                    if (
                        k < (flanking_base_number + maximum_variant_length) or
                        sum(X[row_index, k, :, CountIndex.deletion]) >= (
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

        if inferred_indel_length > 0 and inferred_indel_length < flanking_base_number:
            info.append("LENGUESS=%d" % inferred_indel_length)

        information_string = ""
        if len(info) == 0:
            information_string = "."
        else:
            information_string = ";".join(info)

        genotype_string = ""
        if is_reference:
            genotype_string = "0/0"
        elif is_heterozygous:
            genotype_string = "0/1"
        elif is_homozygous:
            genotype_string = "1/1"

        # if row_index % 10000 == 0:
        #     logging.info("LOG")

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
    z = m.predictZygosityRTVal
    t = m.predictVarTypeRTVal
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
            threadPool.append(Thread(target=Output, args=(args, call_fh, num, XBatch, posBatch, base, z, t, l, )))
            for t in threadPool:
                t.start()
            if end2 == 0:
                end3, num3, XBatch3, posBatch3 = next(tensorGenerator)
            for t in threadPool:
                t.join()
            base = m.predictBaseRTVal
            z = m.predictZygosityRTVal
            t = m.predictVarTypeRTVal
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
        Output(args, call_fh, num2, XBatch2, posBatch2, base, z, t, l)

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
