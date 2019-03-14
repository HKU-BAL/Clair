import sys
import os
import time
import argparse
import param
import logging
import numpy as np
from threading import Thread
from math import log, e
from enum import IntEnum
from collections import namedtuple

import utils
import clair as cv
from utils import BaseChangeIndex, base_change_label_from, GenotypeIndex, genotype_string_from

logging.basicConfig(format='%(message)s', level=logging.INFO)
num2base = dict(zip((0, 1, 2, 3), "ACGT"))
base2num = dict(zip("ACGT", (0, 1, 2, 3)))
v1Type2Name = dict(zip((0, 1, 2, 3, 4), ('HET', 'HOM', 'INS', 'DEL', 'REF')))
v2Zygosity2Name = dict(zip((0, 1), ('HET', 'HOM')))
v2Type2Name = dict(zip((0, 1, 2, 3), ('REF', 'SNP', 'INS', 'DEL')))
v2Length2Name = dict(zip((0, 1, 2, 3, 4, 5), ('0', '1', '2', '3', '4', '4+')))
maximum_variant_length = param.flankingBaseNum  # 5
inferred_indel_length_minimum_allele_frequency = 0.125

Predictions = namedtuple('Predictions', ['base_change_index', 'genotype_index', 'variant_lengths'])


class Channel(IntEnum):
    reference = 0
    insert = 1
    delete = 2
    SNP = 3


def is_reference_from(prediction):
    is_genotype_match = prediction.genotype_index == GenotypeIndex.homo_reference
    return is_genotype_match


def is_homo_SNP_from(prediction):
    is_genotype_match = prediction.genotype_index == GenotypeIndex.homo_variant
    is_base_change_match = (
        prediction.base_change_index == BaseChangeIndex.AA or
        prediction.base_change_index == BaseChangeIndex.CC or
        prediction.base_change_index == BaseChangeIndex.GG or
        prediction.base_change_index == BaseChangeIndex.TT
    )
    is_variant_length_match = prediction.variant_lengths[0] == 0 and prediction.variant_lengths[1] == 0
    votes = (
        (1 if is_base_change_match else 0) +
        (1 if is_genotype_match else 0) +
        (1 if is_variant_length_match else 0)
    )
    return votes >= 2


def is_hetero_SNP_from(prediction):
    is_genotype_match = (
        prediction.genotype_index == GenotypeIndex.hetero_variant or
        prediction.genotype_index == GenotypeIndex.hetero_variant_multi
    )
    is_base_change_match = (
        prediction.base_change_index == BaseChangeIndex.AC or
        prediction.base_change_index == BaseChangeIndex.AG or
        prediction.base_change_index == BaseChangeIndex.AT or
        prediction.base_change_index == BaseChangeIndex.CG or
        prediction.base_change_index == BaseChangeIndex.CT or
        prediction.base_change_index == BaseChangeIndex.GT
    )
    is_variant_length_match = prediction.variant_lengths[0] == 0 and prediction.variant_lengths[1] == 0
    votes = (
        (1 if is_genotype_match else 0) +
        (1 if is_base_change_match else 0) +
        (1 if is_variant_length_match else 0)
    )
    return votes >= 2


def is_homo_insertion_from(prediction):
    is_genotype_match = prediction.genotype_index == GenotypeIndex.homo_variant
    is_base_change_match = prediction.base_change_index == BaseChangeIndex.InsIns
    is_variant_length_match = (
        prediction.variant_lengths[0] > 0 and
        prediction.variant_lengths[1] > 0 and
        prediction.variant_lengths[0] == prediction.variant_lengths[1]
    )
    votes = (
        (1 if is_genotype_match else 0) +
        (1 if is_base_change_match else 0) +
        (1 if is_variant_length_match else 0)
    )
    return votes >= 3


def is_hetero_insertion_from(prediction):
    is_genotype_match = (
        prediction.genotype_index == GenotypeIndex.hetero_variant or
        prediction.genotype_index == GenotypeIndex.hetero_variant_multi
    )
    is_base_change_match = (
        prediction.base_change_index == BaseChangeIndex.InsIns or
        prediction.base_change_index == BaseChangeIndex.AIns or
        prediction.base_change_index == BaseChangeIndex.CIns or
        prediction.base_change_index == BaseChangeIndex.GIns or
        prediction.base_change_index == BaseChangeIndex.TIns
    )
    is_variant_length_match = prediction.variant_lengths[0] >= 0 and prediction.variant_lengths[1] > 0
    votes = (
        (1 if is_genotype_match else 0) +
        (1 if is_base_change_match else 0) +
        (1 if is_variant_length_match else 0)
    )
    return votes >= 3


def is_homo_deletion_from(prediction):
    is_genotype_match = prediction.genotype_index == GenotypeIndex.homo_variant
    is_base_change_match = prediction.base_change_index == BaseChangeIndex.DelDel
    is_variant_length_match = (
        prediction.variant_lengths[0] < 0 and
        prediction.variant_lengths[1] < 0 and
        prediction.variant_lengths[0] == prediction.variant_lengths[1]
    )
    votes = (
        (1 if is_genotype_match else 0) +
        (1 if is_base_change_match else 0) +
        (1 if is_variant_length_match else 0)
    )
    return votes >= 3


def is_hetero_deletion_from(prediction):
    is_genotype_match = (
        prediction.genotype_index == GenotypeIndex.hetero_variant or
        prediction.genotype_index == GenotypeIndex.hetero_variant_multi
    )
    is_base_change_match = (
        prediction.base_change_index == BaseChangeIndex.DelDel or
        prediction.base_change_index == BaseChangeIndex.ADel or
        prediction.base_change_index == BaseChangeIndex.CDel or
        prediction.base_change_index == BaseChangeIndex.GDel or
        prediction.base_change_index == BaseChangeIndex.TDel
    )
    is_variant_length_match = prediction.variant_lengths[0] < 0 and prediction.variant_lengths[1] <= 0
    votes = (
        (1 if is_genotype_match else 0) +
        (1 if is_base_change_match else 0) +
        (1 if is_variant_length_match else 0)
    )
    return votes >= 3


def is_insertion_and_deletion_from(prediction):
    is_genotype_match = (
        prediction.genotype_index == GenotypeIndex.hetero_variant or
        prediction.genotype_index == GenotypeIndex.hetero_variant_multi
    )
    is_base_change_match = prediction.base_change_index == BaseChangeIndex.InsDel
    is_variant_length_match = prediction.variant_lengths[0] < 0 and prediction.variant_lengths[1] > 0
    votes = (
        (1 if is_genotype_match else 0) +
        (1 if is_base_change_match else 0) +
        (1 if is_variant_length_match else 0)
    )
    return votes >= 3


def homo_SNP_bases_from(base_change_probabilities):
    output_bases_probabilities = np.array([
        base_change_probabilities[BaseChangeIndex.AA],
        base_change_probabilities[BaseChangeIndex.CC],
        base_change_probabilities[BaseChangeIndex.GG],
        base_change_probabilities[BaseChangeIndex.TT],
    ])
    output_bases = [
        base_change_label_from(BaseChangeIndex.AA),
        base_change_label_from(BaseChangeIndex.CC),
        base_change_label_from(BaseChangeIndex.GG),
        base_change_label_from(BaseChangeIndex.TT)
    ][np.argmax(output_bases_probabilities)]
    return output_bases[0], output_bases[1]


def hetero_SNP_bases_from(base_change_probabilities):
    output_bases_probabilities = np.array([
        base_change_probabilities[BaseChangeIndex.AC],
        base_change_probabilities[BaseChangeIndex.AG],
        base_change_probabilities[BaseChangeIndex.AT],
        base_change_probabilities[BaseChangeIndex.CG],
        base_change_probabilities[BaseChangeIndex.CT],
        base_change_probabilities[BaseChangeIndex.GT]
    ])
    output_bases = [
        base_change_label_from(BaseChangeIndex.AC),
        base_change_label_from(BaseChangeIndex.AG),
        base_change_label_from(BaseChangeIndex.AT),
        base_change_label_from(BaseChangeIndex.CG),
        base_change_label_from(BaseChangeIndex.CT),
        base_change_label_from(BaseChangeIndex.GT)
    ][np.argmax(output_bases_probabilities)]
    return output_bases[0], output_bases[1]


def hetero_insert_base_from(base_change_probabilities):
    output_bases_probabilities = np.array([
        base_change_probabilities[BaseChangeIndex.AIns],
        base_change_probabilities[BaseChangeIndex.CIns],
        base_change_probabilities[BaseChangeIndex.GIns],
        base_change_probabilities[BaseChangeIndex.TIns]
    ])
    output_bases = [
        base_change_label_from(BaseChangeIndex.AIns),
        base_change_label_from(BaseChangeIndex.CIns),
        base_change_label_from(BaseChangeIndex.GIns),
        base_change_label_from(BaseChangeIndex.TIns)
    ][np.argmax(output_bases_probabilities)]
    return output_bases[0]


def hetero_delete_base_from(base_change_probabilities):
    output_bases_probabilities = np.array([
        base_change_probabilities[BaseChangeIndex.ADel],
        base_change_probabilities[BaseChangeIndex.CDel],
        base_change_probabilities[BaseChangeIndex.GDel],
        base_change_probabilities[BaseChangeIndex.TDel]
    ])
    output_bases = [
        base_change_label_from(BaseChangeIndex.ADel),
        base_change_label_from(BaseChangeIndex.CDel),
        base_change_label_from(BaseChangeIndex.GDel),
        base_change_label_from(BaseChangeIndex.TDel)
    ][np.argmax(output_bases_probabilities)]
    return output_bases[0]


def quality_score_from(base_change_probabilities, genotype_probabilities):
    sorted_base_change_probabilities = np.sort(base_change_probabilities)[::-1]
    sorted_genotype_probabilities = np.sort(genotype_probabilities)[::-1]
    return min(
        int(
            (-10 * log(e, 10)) * log(
                (
                    sorted_base_change_probabilities[1] ** 1.0 *
                    sorted_genotype_probabilities[1] ** 1.0 + 1e-300
                ) /
                (
                    sorted_base_change_probabilities[0] *
                    sorted_genotype_probabilities[0] + 1e-300
                )
            )
        ),
        999
    )


def filtration_value_from(quality_score_for_pass, quality_score):
    if quality_score_for_pass is None:
        return "."
    if quality_score >= quality_score_for_pass:
        return "PASS"
    return "LowQual"


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
    genotype_probabilities,
    variant_length_list,
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
        prediction = Predictions(
            base_change_index=np.argmax(base_change_probabilities[row_index]),
            genotype_index=np.argmax(genotype_probabilities[row_index]),
            variant_lengths=[int(round(variant_length_list[0])), int(round(variant_length_list[1]))].sort()
        )

        is_reference = is_reference_from(prediction)
        if not is_show_reference and is_reference:
            continue

        is_homo_SNP = is_homo_SNP_from(prediction)
        is_hetero_SNP = is_hetero_SNP_from(prediction)
        is_homo_insertion = is_homo_insertion_from(prediction)
        is_hetero_insertion = is_hetero_insertion_from(prediction)
        is_homo_deletion = is_homo_deletion_from(prediction)
        is_hetero_deletion = is_hetero_deletion_from(prediction)
        is_insertion_and_deletion = is_insertion_and_deletion_from(prediction)

        is_SNP = is_homo_SNP or is_hetero_SNP
        is_insertion = is_homo_insertion or is_hetero_insertion
        is_deletion = is_homo_deletion or is_hetero_deletion

        # get chromosome, position and reference bases
        # with flanking "flanking_base_number" flanking bases at position
        chromosome, position, reference_sequence = posBatch[row_index].split(":")
        position = int(position)

        # quality score
        quality_score = quality_score_from(base_change_probabilities, genotype_probabilities)

        # filtration value
        filtration_value = filtration_value_from(quality_score_for_pass=args.qual, quality_score=quality_score)

        # Initialize other variables
        inferred_indel_length = 0
        info = []

        # read depth
        read_depth = 0
        if is_SNP or is_reference:
            read_depth = sum(
                X[row_index, position_center, :, Channel.reference] +
                X[row_index, position_center, :, Channel.SNP]
            )
        elif is_insertion:
            read_depth = sum(
                X[row_index, position_center+1, :, Channel.reference] +
                X[row_index, position_center+1, :, Channel.insert]
            )
        elif is_deletion:
            read_depth = sum(
                X[row_index, position_center+1, :, Channel.reference] +
                X[row_index, position_center+1, :, Channel.delete]
            )
        if read_depth == 0:
            continue

        # geno type string, would changed to 1/2 later if is multi
        if is_reference:
            genotype_string = genotype_string_from(GenotypeIndex.homo_reference)
        elif is_homo_SNP or is_homo_insertion or is_homo_deletion:
            genotype_string = genotype_string_from(GenotypeIndex.homo_variant)
        elif is_hetero_SNP or is_hetero_deletion or is_hetero_insertion or is_insertion_and_deletion:
            genotype_string = genotype_string_from(GenotypeIndex.hetero_variant)

        # reference base and alternate base
        reference_base = ""
        alternate_base = ""
        if is_reference:
            reference_base = reference_sequence[position_center]
            alternate_base = reference_base

        elif is_homo_SNP:
            base1, base2 = homo_SNP_bases_from(base_change_probabilities)
            reference_base = reference_sequence[position_center]
            alternate_base = base1 if base1 != reference_base else base2

        elif is_hetero_SNP:
            base1, base2 = hetero_SNP_bases_from(base_change_probabilities)
            reference_base = reference_sequence[position_center]
            is_multi = base1 != reference_base and base2 != reference_base
            if is_multi:
                alternate_base = "{},{}".format(base1, base2)
                genotype_string = genotype_string_from(GenotypeIndex.hetero_variant_multi)
            else:
                alternate_base = base1 if base1 != reference_base else base2

        elif is_insertion:
            if is_homo_insertion:
                variant_length_1 = 1 if prediction.variant_lengths[0] <= 0 else prediction.variant_lengths[0]
                variant_length_2 = 1 if prediction.variant_lengths[1] <= 0 else prediction.variant_lengths[1]
                variant_length = min(variant_length_1, variant_length_2)
            elif is_hetero_insertion:
                variant_length_1 = 0 if prediction.variant_lengths[0] <= 0 else prediction.variant_lengths[0]
                variant_length_2 = 0 if prediction.variant_lengths[1] <= 0 else prediction.variant_lengths[1]
                variant_length = max(variant_length_1, variant_length_2)

            if is_hetero_insertion and variant_length <= 0:
                continue

            if variant_length_2 < variant_length_1:
                variant_length_1, variant_length_2 = variant_length_2, variant_length_1

            reference_base = reference_sequence[position_center]

            is_inferred_variant_length = variant_length >= maximum_variant_length
            if is_inferred_variant_length:
                for k in range(flanking_base_number + 1, 2 * flanking_base_number + 1):
                    reference_tensor = X[row_index, k, :, Channel.reference]
                    insertion_tensor = X[row_index, k, :, Channel.insert]
                    if (
                        k < (flanking_base_number + maximum_variant_length) or
                        sum(insertion_tensor) >= inferred_indel_length_minimum_allele_frequency * sum(reference_tensor)
                    ):
                        inferred_indel_length += 1
                        alternate_base += num2base[np.argmax(insertion_tensor) % 4]
                    else:
                        break
            else:
                for k in range(flanking_base_number + 1, flanking_base_number + variant_length + 1):
                    alternate_base += num2base[np.argmax(X[row_index, k, :, Channel.insert]) % 4]

            is_marked_as_SV = is_inferred_variant_length and inferred_indel_length >= flanking_base_number
            hetero_insert_base = hetero_insert_base_from(base_change_probabilities)
            is_SNP_Ins_multi = (
                not is_marked_as_SV and
                is_hetero_insertion and
                (
                    prediction.base_change_index == BaseChangeIndex.AIns or
                    prediction.base_change_index == BaseChangeIndex.CIns or
                    prediction.base_change_index == BaseChangeIndex.GIns or
                    prediction.base_change_index == BaseChangeIndex.TIns
                ) and
                hetero_insert_base != reference_base
            )
            is_Ins_Ins_multi = (
                not is_marked_as_SV and
                is_hetero_insertion and
                prediction.base_change_index == BaseChangeIndex.InsIns and
                variant_length_1 > 0 and variant_length_2 > 0 and
                variant_length_1 != variant_length_2
            )

            if is_marked_as_SV:
                alternate_base = "<INS>"
                info.append("SVTYPE=INS")
            else:
                alternate_base = reference_base + alternate_base

            if is_SNP_Ins_multi:
                alternate_base = "{},{}".format(hetero_insert_base, alternate_base)
                genotype_string = genotype_string_from(GenotypeIndex.hetero_variant_multi)
            elif is_Ins_Ins_multi:
                alternate_base_1 = alternate_base[0:len(reference_base) + variant_length_1]
                alternate_base_2 = alternate_base
                if alternate_base_1 != alternate_base_2:
                    alternate_base = "{},{}".format(alternate_base_1, alternate_base_2)
                    genotype_string = genotype_string_from(GenotypeIndex.hetero_variant_multi)

        elif is_deletion:
            if is_homo_deletion:
                variant_length_1 = 1 if prediction.variant_lengths[0] >= 0 else -prediction.variant_lengths[0]
                variant_length_2 = 1 if prediction.variant_lengths[1] >= 0 else -prediction.variant_lengths[1]
                variant_length = min(variant_length_1, variant_length_2)
            elif is_hetero_deletion:
                variant_length_1 = 0 if prediction.variant_lengths[0] >= 0 else -prediction.variant_lengths[0]
                variant_length_2 = 0 if prediction.variant_lengths[1] >= 0 else -prediction.variant_lengths[1]
                variant_length = max(variant_length_1, variant_length_2)

            if is_hetero_deletion and variant_length >= 0:
                continue

            if variant_length_2 < variant_length_1:
                variant_length_1, variant_length_2 = variant_length_2, variant_length_1

            is_inferred_variant_length = variant_length >= maximum_variant_length
            if is_inferred_variant_length:
                for k in range(flanking_base_number + 1, 2 * flanking_base_number + 1):
                    reference_tensor = X[row_index, k, :, Channel.reference]
                    deletion_tensor = X[row_index, k, :, Channel.delete]
                    if (
                        k < (flanking_base_number + maximum_variant_length) or
                        sum(reference_tensor) >= inferred_indel_length_minimum_allele_frequency * sum(deletion_tensor)
                    ):
                        inferred_indel_length += 1
                    else:
                        break

            is_marked_as_SV = is_inferred_variant_length and inferred_indel_length >= flanking_base_number
            hetero_delete_base = hetero_delete_base_from(base_change_probabilities)
            is_SNP_Del_multi = (
                not is_marked_as_SV and
                is_hetero_deletion and
                (
                    prediction.base_change_index == BaseChangeIndex.ADel or
                    prediction.base_change_index == BaseChangeIndex.CDel or
                    prediction.base_change_index == BaseChangeIndex.GDel or
                    prediction.base_change_index == BaseChangeIndex.TDel
                ) and
                hetero_delete_base != reference_base
            )
            is_Del_Del_multi = (
                not is_marked_as_SV and
                is_hetero_deletion and
                prediction.base_change_index == BaseChangeIndex.DelDel and
                variant_length_1 > 0 and variant_length_2 > 0 and
                variant_length_1 != variant_length_2
            )

            if is_marked_as_SV:
                reference_base = reference_sequence[position_center]
                alternate_base = "<DEL>"
                info.append("SVTYPE=DEL")
            elif is_inferred_variant_length:
                reference_base = reference_sequence[position_center:position_center + inferred_indel_length + 1]
                alternate_base = reference_sequence[position_center]
            else:
                reference_base = reference_sequence[position_center:position_center + variant_length + 1]
                alternate_base = reference_sequence[position_center]

            if is_SNP_Del_multi:
                alternate_base_1 = alternate_base
                alternate_base_2 = hetero_delete_base + reference_base[1:]
                alternate_base = "{},{}".format(alternate_base_1, alternate_base_2)
                genotype_string = genotype_string_from(GenotypeIndex.hetero_variant_multi)
            elif is_Del_Del_multi:
                alternate_base_1 = alternate_base
                alternate_base_2 = reference_sequence[position_center:position_center + variant_length_2 - variant_length_1 + 1]
                if alternate_base_1 != alternate_base_2:
                    alternate_base = "{},{}".format(alternate_base_1, alternate_base_2)
                    genotype_string = genotype_string_from(GenotypeIndex.hetero_variant_multi)


        elif is_insertion_and_deletion:
            # TODO
            continue

        # allele frequency / supported reads
        supported_reads_count = 0
        if is_reference:
            supported_reads_count = (
                X[row_index, position_center,   base2num[reference_base], Channel.reference] +
                X[row_index, position_center, base2num[reference_base]+4, Channel.reference]
            )
        elif is_SNP:
            for base in alternate_base:
                if base == ',':
                    continue
                supported_reads_count += (
                    X[row_index, position_center,   base2num[base], Channel.SNP] +
                    X[row_index, position_center, base2num[base]+4, Channel.SNP]
                )
        elif is_insertion:
            supported_reads_count = sum(X[row_index, position_center+1, :, Channel.insert])
        elif is_deletion:
            supported_reads_count = sum(X[row_index, position_center+1, :, Channel.delete])
        allele_frequency = ((supported_reads_count + 0.0) / read_depth) if read_depth != 0 else 0.0

        # if using inferred indel length, add info LENGUESS
        if 0 < inferred_indel_length < flanking_base_number:
            info.append("LENGUESS={}".format(inferred_indel_length))

        # information string
        information_string = ""
        if len(info) == 0:
            information_string = "."
        else:
            information_string = ";".join(info)

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
            allele_frequency
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
        summaries = m.get_activation_summary(
            batch_X,
            operations=m.layers,
            batch_item_suffixes=batch_positions,
            max_plot_in_batch=args.max_plot - num_plotted if args.max_plot >= 0 else batch_size,
            parallel_level=args.parallel_level,
            num_workers=args.workers,
            fast_plotting=args.fast_plotting
        )
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
    gt = m.predictGenotypeRTVal
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
            gt = m.predictGenotypeRTVal
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
