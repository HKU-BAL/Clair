import os
import sys
import argparse
import re
import shlex
import subprocess
import gc
import signal
import param
import intervaltree
import random
from math import log

is_pypy = '__pypy__' in sys.builtin_module_names

RATIO_OF_NON_VARIANT_TO_VARIANT = 2.0


def PypyGCCollect(signum, frame):
    gc.collect()
    signal.alarm(60)


def variants_map_from(variant_file_path):
    """
        variants map with 1-based position as key
    """
    if variant_file_path == None:
        return {}

    variants_map = {}
    f = subprocess.Popen(shlex.split("gzip -fdc %s" % (variant_file_path)), stdout=subprocess.PIPE, bufsize=8388608)

    while True:
        row = f.stdout.readline()
        is_finish_reading_output = row == '' and f.poll() is not None
        if is_finish_reading_output:
            break

        if row:
            columns = row.split()
            ctg_name, position_str = columns[0], columns[1]
            key = ctg_name + ":" + position_str

            variants_map[key] = True

    f.stdout.close()
    f.wait()

    return variants_map


def non_variants_map_near_variants_from(variants_map, lower_limit_to_non_variants=15, upper_limit_to_non_variants=16):
    """
        non variants map with 1-based position as key
    """
    non_variants_map = {}
    non_variants_map_to_exclude = {}

    for key in variants_map.keys():
        ctg_name, position_str = key.split(':')
        position = int(position_str)

        for i in range(upper_limit_to_non_variants * 2 + 1):
            position_offset = -upper_limit_to_non_variants + i
            temp_position = position + position_offset
            if temp_position <= 0:
                continue

            temp_key = ctg_name + ":" + str(temp_position)
            can_add_to_non_variants_map = (
                temp_key not in variants_map and
                temp_key not in non_variants_map and
                (
                    -upper_limit_to_non_variants <= position_offset <= -lower_limit_to_non_variants or
                    lower_limit_to_non_variants <= position_offset <= upper_limit_to_non_variants
                )
            )
            can_add_to_non_variants_map_to_exclude = (
                lower_limit_to_non_variants > position_offset > -lower_limit_to_non_variants
            )
            if can_add_to_non_variants_map:
                non_variants_map[temp_key] = True
            if can_add_to_non_variants_map_to_exclude:
                non_variants_map_to_exclude[temp_key] = True

    for key in non_variants_map_to_exclude.keys():
        if key in non_variants_map:
            del non_variants_map[key]

    return non_variants_map


class CandidateStdout(object):
    def __init__(self, handle):
        self.stdin = handle

    def __del__(self):
        self.stdin.close()


def region_from(ctg_name, ctg_start=None, ctg_end=None):
    if ctg_name is None:
        return ""
    if (ctg_start is None) != (ctg_end is None):
        return ""

    if ctg_start is None and ctg_end is None:
        return "{}".format(ctg_name)
    return "{}:{}-{}".format(ctg_name, ctg_start, ctg_end)


def reference_sequence_from(samtools_execute_command, fasta_file_path, regions):
    refernce_sequences = []
    region_value_for_faidx = " ".join(regions)

    samtools_faidx_process = subprocess.Popen(
        shlex.split("{} faidx {} {}".format(samtools_execute_command, fasta_file_path, region_value_for_faidx)),
        stdout=subprocess.PIPE,
        bufsize=8388608
    )
    while True:
        row = samtools_faidx_process.stdout.readline()
        is_finish_reading_output = row == '' and samtools_faidx_process.poll() is not None
        if is_finish_reading_output:
            break
        if row:
            refernce_sequences.append(row.rstrip())

    # first line is reference name ">xxxx", need to be ignored
    reference_sequence = "".join(refernce_sequences[1:])

    samtools_faidx_process.stdout.close()
    samtools_faidx_process.wait()
    if samtools_faidx_process.returncode != 0:
        return None

    return reference_sequence


def interval_tree_map_from(bed_file_path):
    interval_tree_map = {}
    if bed_file_path is None:
        return interval_tree_map

    gzip_process = subprocess.Popen(
        shlex.split("gzip -fdc %s" % (bed_file_path)),
        stdout=subprocess.PIPE,
        bufsize=8388608
    )

    while True:
        row = gzip_process.stdout.readline()
        is_finish_reading_output = row == '' and gzip_process.poll() is not None
        if is_finish_reading_output:
            break

        if row:
            columns = row.strip().split()

            ctg_name = columns[0]
            if ctg_name not in interval_tree_map:
                interval_tree_map[ctg_name] = intervaltree.IntervalTree()

            ctg_start = int(columns[1])
            ctg_end = int(columns[2])-1
            if ctg_start == ctg_end:
                ctg_end += 1

            interval_tree_map[ctg_name].addi(ctg_start, ctg_end)

    gzip_process.stdout.close()
    gzip_process.wait()

    return interval_tree_map


def is_too_many_soft_clipped_bases_for_a_read_from(CIGAR):
    soft_clipped_bases = 0
    total_alignment_positions = 0

    advance = 0
    for c in str(CIGAR):
        if c.isdigit():
            advance = advance * 10 + int(c)
            continue
        if c == "S":
            soft_clipped_bases += advance
        total_alignment_positions += advance
        advance = 0

    # skip a read less than 55% aligned
    return 1.0 - float(soft_clipped_bases) / (total_alignment_positions + 1) < 0.55


def make_candidates(args):

    gen4Training = args.gen4Training
    variant_file_path = args.var_fn
    bed_file_path = args.bed_fn
    fasta_file_path = args.ref_fn
    ctg_name = args.ctgName
    ctg_start = args.ctgStart
    ctg_end = args.ctgEnd
    output_probability = args.outputProb
    samtools_execute_command = args.samtools
    minimum_depth_for_candidate = args.minCoverage
    minimum_af_for_candidate = args.threshold
    minimum_mapping_quality = args.minMQ
    bam_file_path = args.bam_fn
    candidate_output_path = args.can_fn
    is_using_stdout_for_output_candidate = candidate_output_path == "PIPE"

    is_building_training_dataset = gen4Training == True
    is_variant_file_given = variant_file_path is not None
    is_bed_file_given = bed_file_path is not None
    is_ctg_name_given = ctg_name is not None
    is_ctg_range_given = is_ctg_name_given and ctg_start is not None and ctg_end is not None

    if is_building_training_dataset:
        # minimum_depth_for_candidate = 0
        minimum_af_for_candidate = 0

    # preparation for candidates near variants
    need_consider_candidates_near_variant = is_building_training_dataset and is_variant_file_given
    variants_map = variants_map_from(variant_file_path) if need_consider_candidates_near_variant else {}
    non_variants_map = non_variants_map_near_variants_from(variants_map)
    no_of_candidates_near_variant = 0
    no_of_candidates_outside_variant = 0

    # update output probabilities for candidates near variants
    # original: (7000000.0 * 2.0 / 3000000000)
    ratio_of_candidates_near_variant_to_candidates_outside_variant = 1.0
    output_probability_near_variant = (
        3500000.0 * ratio_of_candidates_near_variant_to_candidates_outside_variant * RATIO_OF_NON_VARIANT_TO_VARIANT / 14000000
    )
    output_probability_outside_variant = 3500000.0 * RATIO_OF_NON_VARIANT_TO_VARIANT / (3000000000 - 14000000)

    if not os.path.isfile("{}.fai".format(fasta_file_path)):
        print >> sys.stderr, "Fasta index {}.fai doesn't exist.".format(fasta_file_path)
        sys.exit(1)

    regions = []
    reference_start = None
    reference_end = None
    if is_ctg_range_given:
        ctg_start += 1
        reference_start = ctg_start
        reference_end = ctg_end

        reference_start -= param.expandReferenceRegion
        reference_start = 1 if reference_start < 1 else reference_start
        reference_end += param.expandReferenceRegion

        regions.append(region_from(ctg_name=ctg_name, ctg_start=reference_start, ctg_end=reference_end))
    elif is_ctg_name_given:
        regions.append(region_from(ctg_name=ctg_name))

    reference_sequence = reference_sequence_from(
        samtools_execute_command=samtools_execute_command,
        fasta_file_path=fasta_file_path,
        regions=regions
    )
    if reference_sequence is None or len(reference_sequence) == 0:
        print >> sys.stderr, "[ERROR] Failed to load reference seqeunce from file ({}).".format(fasta_file_path)
        sys.exit(1)

    tree = interval_tree_map_from(bed_file_path=bed_file_path)
    if is_bed_file_given and ctg_name not in tree:
        print >> sys.stderr, "[ERROR] ctgName not exists in bed file ({}).".format(bed_file_path)
        sys.exit(1)

    samtools_view_process = subprocess.Popen(
        shlex.split("{} view -F 2308 {} {}".format(samtools_execute_command, bam_file_path, " ".join(regions))),
        stdout=subprocess.PIPE,
        bufsize=8388608
    )

    if is_using_stdout_for_output_candidate:
        can_fp = CandidateStdout(sys.stdout)
    else:
        can_fpo = open(candidate_output_path, "wb")
        can_fp = subprocess.Popen(
            shlex.split("gzip -c"), stdin=subprocess.PIPE, stdout=can_fpo, stderr=sys.stderr, bufsize=8388608
        )

    pileup = {}
    zero_based_position = 0
    POS = 0
    number_of_reads_processed = 0

    while True:
        row = samtools_view_process.stdout.readline()
        is_finish_reading_output = row == '' and samtools_view_process.poll() is not None

        if row:
            columns = row.strip().split()
            if columns[0][0] == "@":
                continue

            RNAME = columns[2]
            if RNAME != ctg_name:
                continue

            POS = int(columns[3]) - 1  # switch from 1-base to 0-base to match sequence index
            MAPQ = int(columns[4])
            CIGAR = columns[5]
            SEQ = columns[9]

            reference_position = POS
            query_position = 0

            if MAPQ < minimum_mapping_quality:
                continue
            if CIGAR == "*" or is_too_many_soft_clipped_bases_for_a_read_from(CIGAR):
                continue

            number_of_reads_processed += 1

            advance = 0
            for c in str(CIGAR):
                if c.isdigit():
                    advance = advance * 10 + int(c)
                    continue

                if c == "S":
                    query_position += advance

                elif c == "M" or c == "=" or c == "X":
                    for _ in range(advance):
                        pos = reference_position
                        base = SEQ[query_position]

                        pileup.setdefault(pos, {"A": 0, "C": 0, "G": 0, "T": 0, "I": 0, "D": 0, "N": 0})
                        pileup[pos][base] += 1

                        # those CIGAR operations consumes query and reference
                        reference_position += 1
                        query_position += 1

                elif c == "I":
                    pileup.setdefault(reference_position - 1, {"A": 0, "C": 0, "G": 0, "T": 0, "I": 0, "D": 0, "N": 0})
                    pileup[reference_position - 1]["I"] += 1

                    # insertion consumes query
                    query_position += advance

                elif c == "D":
                    pileup.setdefault(reference_position - 1, {"A": 0, "C": 0, "G": 0, "T": 0, "I": 0, "D": 0, "N": 0})
                    pileup[reference_position - 1]["D"] += 1

                    # deletion consumes reference
                    reference_position += advance

                # reset advance
                advance = 0

        positions = list(filter(lambda x: x < POS, pileup.keys())) if not is_finish_reading_output else pileup.keys()
        positions.sort()
        for zero_based_position in positions:
            baseCount = depth = reference_base = temp_key = None

            # ctg and bed checking
            pass_ctg = not is_ctg_range_given or ctg_start <= zero_based_position <= ctg_end
            pass_bed = not is_bed_file_given or (ctg_name in tree and len(tree[ctg_name].search(zero_based_position)) != 0)

            # output probability checking
            pass_output_probability = True
            if pass_ctg and pass_bed and is_building_training_dataset and is_variant_file_given:
                temp_key = ctg_name + ":" + str(zero_based_position+1)
                pass_output_probability = (
                    temp_key not in variants_map and (
                        (temp_key in non_variants_map and random.uniform(0, 1) <= output_probability_near_variant) or
                        (temp_key not in non_variants_map and random.uniform(0, 1) <= output_probability_outside_variant)
                    )
                )
            elif pass_ctg and pass_bed and is_building_training_dataset:
                pass_output_probability = random.uniform(0, 1) <= output_probability

            # depth checking
            pass_depth = False
            if pass_ctg and pass_bed and pass_output_probability:
                baseCount = pileup[zero_based_position].items()
                depth = sum(x[1] for x in baseCount)
                pass_depth = depth >= minimum_depth_for_candidate

            # af checking
            pass_af = False
            if pass_ctg and pass_bed and pass_output_probability and pass_depth:
                reference_base = reference_sequence[zero_based_position - (0 if reference_start is None else (reference_start - 1))]
                denominator = depth if depth > 0 else 1
                baseCount.sort(key=lambda x: -x[1])  # sort baseCount descendingly
                p0, p1 = float(baseCount[0][1]) / denominator, float(baseCount[1][1]) / denominator
                pass_af = (
                    (p0 <= 1.0 - minimum_af_for_candidate and p1 >= minimum_af_for_candidate) or
                    baseCount[0][0] != reference_base
                )

            # output
            need_output_candidate = pass_ctg and pass_bed and pass_output_probability and pass_depth and pass_af
            if need_output_candidate:
                if temp_key is not None and temp_key in non_variants_map:
                    no_of_candidates_near_variant += 1
                elif temp_key is not None and temp_key not in non_variants_map:
                    no_of_candidates_outside_variant += 1

                output = [ctg_name, zero_based_position+1, reference_base, depth]
                output.extend(["%s %d" % x for x in baseCount])
                output = " ".join([str(x) for x in output]) + "\n"

                can_fp.stdin.write(output)

        for zero_based_position in positions:
            del pileup[zero_based_position]

        if is_finish_reading_output:
            break

    if need_consider_candidates_near_variant:
        print "# of candidates near variant: ", no_of_candidates_near_variant
        print "# of candidates outside variant: ", no_of_candidates_outside_variant

    samtools_view_process.stdout.close()
    samtools_view_process.wait()

    if not is_using_stdout_for_output_candidate:
        can_fp.stdin.close()
        can_fp.wait()
        can_fpo.close()

    if number_of_reads_processed == 0:
        print >> sys.stderr, "No read has been process, either the genome region you specified has no read cover, or please check the correctness of your BAM input (%s)." % (
            bam_file_path)
        sys.exit(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate variant candidates using alignments")

    parser.add_argument('--bam_fn', type=str, default="input.bam",
                        help="Sorted bam file input, default: %(default)s")

    parser.add_argument('--ref_fn', type=str, default="ref.fa",
                        help="Reference fasta file input, default: %(default)s")

    parser.add_argument('--bed_fn', type=str, default=None,
                        help="Call variant only in these regions, works in intersection with ctgName, ctgStart and ctgEnd, optional, default: as defined by ctgName, ctgStart and ctgEnd")

    parser.add_argument('--can_fn', type=str, default="PIPE",
                        help="Pile-up count output, use PIPE for standard output, default: %(default)s")

    parser.add_argument('--var_fn', type=str, default=None,
                        help="Candidate sites VCF file input, if provided, will choose candidate +/- 1 or +/- 2. Use together with gen4Training. default: %(default)s")

    parser.add_argument('--threshold', type=float, default=0.125,
                        help="Minimum allele frequence of the 1st non-reference allele for a site to be considered as a condidate site, default: %(default)f")

    parser.add_argument('--minCoverage', type=float, default=4,
                        help="Minimum coverage required to call a variant, default: %(default)f")

    parser.add_argument('--minMQ', type=int, default=0,
                        help="Minimum Mapping Quality. Mapping quality lower than the setting will be filtered, default: %(default)d")

    parser.add_argument('--gen4Training', type=param.str2bool, nargs='?', const=True, default=False,
                        help="Output all genome positions as candidate for model training (Set --threshold to 0, --minCoverage to 0), default: %(default)s")

    # parser.add_argument('--candidates', type=int, default=7000000,
    #         help="Use with gen4Training, number of variant candidates to be generated, default: %(default)s")

    # parser.add_argument('--genomeSize', type=int, default=3000000000,
    #         help="Use with gen4Training, default: %(default)s")

    parser.add_argument('--outputProb', type=float, default=(7000000.0 * RATIO_OF_NON_VARIANT_TO_VARIANT / 3000000000),
                        help="output probability")

    parser.add_argument('--ctgName', type=str, default="chr17",
                        help="The name of sequence to be processed, default: %(default)s")

    parser.add_argument('--ctgStart', type=int, default=None,
                        help="The 1-base starting position of the sequence to be processed")

    parser.add_argument('--ctgEnd', type=int, default=None,
                        help="The inclusive ending position of the sequence to be processed")

    parser.add_argument('--samtools', type=str, default="samtools",
                        help="Path to the 'samtools', default: %(default)s")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    make_candidates(args)
