import os
import sys
import argparse
import os
import re
import shlex
import subprocess
import signal
import gc
import shared.param as param
from collections import namedtuple

is_pypy = '__pypy__' in sys.builtin_module_names

ReferenceResult = namedtuple('ReferenceResult', ['name', 'start', 'end', 'sequence', 'is_faidx_process_have_error'])


def PypyGCCollect(signum, frame):
    gc.collect()
    signal.alarm(60)


base2num = dict(zip("ACGT", (0, 1, 2, 3)))
stripe2 = param.matrixRow * param.matrixNum
stripe1 = param.matrixNum


def generate_tensor(ctg_name, alignments, center, reference_sequence, reference_start_0_based, minimum_coverage):
    flanking_base_num = param.flankingBaseNum
    matrix_row = param.matrixRow
    matrix_num = param.matrixNum
    BASES = set("ACGT-")
    NUMBER_OF_POSITIONS = 2 * flanking_base_num + 1

    alignment_code = [0] * (NUMBER_OF_POSITIONS * matrix_row * matrix_num)
    depth = [0] * NUMBER_OF_POSITIONS
    for alignment in alignments:
        for reference_position, queryAdv, reference_base, query_base, STRAND in alignment:
            if str(reference_base) not in BASES or str(query_base) not in BASES:
                continue
            if not (-(flanking_base_num + 1) <= reference_position - center < flanking_base_num):
                continue

            offset = reference_position - center + (flanking_base_num + 1)
            if query_base != "-" and reference_base != "-":
                depth[offset] = depth[offset] + 1
                alignment_code[stripe2*offset + stripe1*(base2num[reference_base] + STRAND*4) + 0] += 1
                alignment_code[stripe2*offset + stripe1*(base2num[query_base] + STRAND*4) + 1] += 1
                alignment_code[stripe2*offset + stripe1*(base2num[reference_base] + STRAND*4) + 2] += 1
                alignment_code[stripe2*offset + stripe1*(base2num[query_base] + STRAND*4) + 3] += 1
            elif query_base != "-" and reference_base == "-":
                idx = min(offset+queryAdv, NUMBER_OF_POSITIONS - 1)
                alignment_code[stripe2*idx + stripe1*(base2num[query_base] + STRAND*4) + 1] += 1
            elif query_base == "-" and reference_base != "-":
                alignment_code[stripe2*offset + stripe1*(base2num[reference_base] + STRAND*4) + 2] += 1
            else:
                print >> sys.stderr, "Should not reach here: %s, %s" % (reference_base, query_base)

    new_reference_position = center - reference_start_0_based
    if new_reference_position - (flanking_base_num+1) < 0 or depth[flanking_base_num] < minimum_coverage:
        return None
    return "%s %d %s %s" % (
        ctg_name,
        center,
        reference_sequence[new_reference_position-(flanking_base_num+1):new_reference_position + flanking_base_num],
        " ".join("%d" % x for x in alignment_code)
    )


def candidate_position_generator_from(
    candidate_file_path,
    ctg_start,
    ctg_end,
    is_consider_left_edge,
    flanking_base_num,
    begin_to_end
):
    is_read_file_from_standard_input = candidate_file_path == "PIPE"
    if is_read_file_from_standard_input:
        candidate_file_path_output = sys.stdin
    else:
        candidate_file_path_process = subprocess.Popen(
            shlex.split("pigz -fdc %s" % (candidate_file_path)), stdout=subprocess.PIPE, bufsize=8388608
        )
        candidate_file_path_output = candidate_file_path_process.stdout

    is_ctg_region_provided = ctg_start is not None and ctg_end is not None

    for row in candidate_file_path_output:
        row = row.split()
        position = int(row[1])  # 1-based position

        if is_ctg_region_provided and not (ctg_start <= position <= ctg_end):
            continue

        if is_consider_left_edge:
            # i is 0-based
            for i in range(position - (flanking_base_num + 1), position + (flanking_base_num + 1)):
                if i not in begin_to_end:
                    begin_to_end[i] = [(position + (flanking_base_num + 1), position)]
                else:
                    begin_to_end[i].append((position + (flanking_base_num + 1), position))
        else:
            begin_to_end[position - (flanking_base_num + 1)] = [(position + (flanking_base_num + 1), position)]

        yield position

    if not is_read_file_from_standard_input:
        candidate_file_path_output.close()
        candidate_file_path_process.wait()
    yield -1


class TensorStdout(object):
    def __init__(self, handle):
        self.stdin = handle

    def __del__(self):
        self.stdin.close()


def reference_result_from(
    ctg_name,
    ctg_start,
    ctg_end,
    samtools,
    reference_file_path,
    expand_reference_region
):
    region_str = ""
    reference_start, reference_end = None, None
    have_start_and_end_positions = ctg_start != None and ctg_end != None
    if have_start_and_end_positions:
        reference_start, reference_end = ctg_start - expand_reference_region, ctg_end + expand_reference_region
        reference_start = 1 if reference_start < 1 else reference_start
        region_str = "%s:%d-%d" % (ctg_name, reference_start, reference_end)
    else:
        region_str = ctg_name

    faidx_process = subprocess.Popen(
        shlex.split("%s faidx %s %s" % (samtools, reference_file_path, region_str)),
        stdout=subprocess.PIPE,
        bufsize=8388608
    )
    if faidx_process is None:
        return None

    reference_name = None
    reference_sequences = []
    for row in faidx_process.stdout:
        if reference_name is None:
            reference_name = row.rstrip().lstrip(">") or ""
        else:
            reference_sequences.append(row.rstrip())
    reference_sequence = "".join(reference_sequences)

    faidx_process.stdout.close()
    faidx_process.wait()

    return ReferenceResult(
        name=reference_name,
        start=reference_start,
        end=reference_end,
        sequence=reference_sequence,
        is_faidx_process_have_error=faidx_process.returncode != 0,
    )


def samtools_view_process_from(
    ctg_name,
    ctg_start,
    ctg_end,
    samtools,
    bam_file_path
):
    have_start_and_end_position = ctg_start != None and ctg_end != None
    region_str = ("%s:%d-%d" % (ctg_name, ctg_start, ctg_end)) if have_start_and_end_position else ctg_name

    return subprocess.Popen(
        shlex.split("%s view -F 2308 %s %s" % (samtools, bam_file_path, region_str)),
        stdout=subprocess.PIPE,
        bufsize=8388608
    )


def OutputAlnTensor(args):
    available_slots = 10000000
    samtools = args.samtools
    tensor_file_path = args.tensor_fn
    bam_file_path = args.bam_fn
    reference_file_path = args.ref_fn
    candidate_file_path = args.can_fn
    dcov = args.dcov
    is_consider_left_edge = not args.stop_consider_left_edge
    min_coverage = args.minCoverage
    minimum_mapping_quality = args.minMQ
    ctg_name = args.ctgName
    ctg_start = args.ctgStart
    ctg_end = args.ctgEnd

    reference_result = reference_result_from(
        ctg_name=ctg_name,
        ctg_start=ctg_start,
        ctg_end=ctg_end,
        samtools=samtools,
        reference_file_path=reference_file_path,
        expand_reference_region=param.expandReferenceRegion,
    )

    reference_sequence = reference_result.sequence if reference_result is not None else ""
    is_faidx_process_have_error = reference_result is None or reference_result.is_faidx_process_have_error
    have_reference_sequence = reference_result is not None and len(reference_sequence) > 0

    if reference_result is None or is_faidx_process_have_error or not have_reference_sequence:
        print >> sys.stderr, "Failed to load reference seqeunce. Please check if the provided reference fasta %s and the ctgName %s are correct." % (
            reference_file_path,
            ctg_name
        )
        sys.exit(1)

    reference_start = reference_result.start
    reference_start_0_based = 0 if reference_start is None else (reference_start - 1)
    begin_to_end = {}
    candidate_position = 0
    candidate_position_generator = candidate_position_generator_from(
        candidate_file_path=candidate_file_path,
        ctg_start=ctg_start,
        ctg_end=ctg_end,
        is_consider_left_edge=is_consider_left_edge,
        flanking_base_num=param.flankingBaseNum,
        begin_to_end=begin_to_end
    )

    samtools_view_process = samtools_view_process_from(
        ctg_name=ctg_name,
        ctg_start=ctg_start,
        ctg_end=ctg_end,
        samtools=samtools,
        bam_file_path=bam_file_path
    )

    center_to_alignment = {}

    if tensor_file_path != "PIPE":
        tensor_fpo = open(tensor_file_path, "wb")
        tensor_fp = subprocess.Popen(
            shlex.split("pigz -c"), stdin=subprocess.PIPE, stdout=tensor_fpo, stderr=sys.stderr, bufsize=8388608
        )
    else:
        tensor_fp = TensorStdout(sys.stdout)

    previous_position = 0
    depthCap = 0
    for l in samtools_view_process.stdout:
        l = l.split()
        if l[0][0] == "@":
            continue

        FLAG = int(l[1])
        POS = int(l[3]) - 1  # switch from 1-base to 0-base to match sequence index
        MQ = int(l[4])
        CIGAR = l[5]
        SEQ = l[9]
        reference_position = POS
        query_position = 0
        STRAND = (16 == (FLAG & 16))

        if MQ < minimum_mapping_quality:
            continue

        end_to_center = {}
        active_set = set()

        while candidate_position != -1 and candidate_position < (POS + len(SEQ) + 100000):
            candidate_position = next(candidate_position_generator)

        if previous_position != POS:
            previous_position = POS
            depthCap = 0
        else:
            depthCap += 1
            if depthCap >= dcov:
                #print >> sys.stderr, "Bypassing POS %d at depth %d\n" % (POS, depthCap)
                continue

        advance = 0
        for c in str(CIGAR):
            if available_slots <= 0:
                break

            if c.isdigit():
                advance = advance * 10 + int(c)
                continue

            # soft clip
            if c == "S":
                query_position += advance

            # match / mismatch
            if c == "M" or c == "=" or c == "X":
                for _ in xrange(advance):
                    if reference_position in begin_to_end:
                        for rEnd, rCenter in begin_to_end[reference_position]:
                            if rCenter in active_set:
                                continue
                            end_to_center[rEnd] = rCenter
                            active_set.add(rCenter)
                            center_to_alignment.setdefault(rCenter, [])
                            center_to_alignment[rCenter].append([])
                    for center in list(active_set):
                        if available_slots <= 0:
                            break
                        available_slots -= 1

                        center_to_alignment[center][-1].append((
                            reference_position,
                            0,
                            reference_sequence[reference_position - reference_start_0_based],
                            SEQ[query_position],
                            STRAND
                        ))
                    if reference_position in end_to_center:
                        center = end_to_center[reference_position]
                        active_set.remove(center)
                    reference_position += 1
                    query_position += 1

            # insertion
            if c == "I":
                for queryAdv in xrange(advance):
                    for center in list(active_set):
                        if available_slots <= 0:
                            break
                        available_slots -= 1

                        center_to_alignment[center][-1].append((
                            reference_position,
                            queryAdv,
                            "-",
                            SEQ[query_position],
                            STRAND
                        ))
                    query_position += 1

            # deletion
            if c == "D":
                for _ in xrange(advance):
                    for center in list(active_set):
                        if available_slots <= 0:
                            break
                        available_slots -= 1

                        center_to_alignment[center][-1].append((
                            reference_position,
                            0,
                            reference_sequence[reference_position - reference_start_0_based],
                            "-",
                            STRAND
                        ))
                    if reference_position in begin_to_end:
                        for rEnd, rCenter in begin_to_end[reference_position]:
                            if rCenter in active_set:
                                continue
                            end_to_center[rEnd] = rCenter
                            active_set.add(rCenter)
                            center_to_alignment.setdefault(rCenter, [])
                            center_to_alignment[rCenter].append([])
                    if reference_position in end_to_center:
                        center = end_to_center[reference_position]
                        active_set.remove(center)
                    reference_position += 1

            # reset advance
            advance = 0

        if depthCap == 0:
            for center in center_to_alignment.keys():
                if center + (param.flankingBaseNum + 1) >= POS:
                    continue
                l = generate_tensor(
                    ctg_name, center_to_alignment[center], center, reference_sequence, reference_start_0_based, min_coverage
                )
                if l != None:
                    tensor_fp.stdin.write(l)
                    tensor_fp.stdin.write("\n")
                available_slots += sum(len(i) for i in center_to_alignment[center])
                #print >> sys.stderr, "POS %d: remaining slots %d" % (center, available_slots)
                del center_to_alignment[center]

    for center in center_to_alignment.keys():
        l = generate_tensor(
            ctg_name, center_to_alignment[center], center, reference_sequence, reference_start_0_based, min_coverage
        )
        if l != None:
            tensor_fp.stdin.write(l)
            tensor_fp.stdin.write("\n")

    samtools_view_process.stdout.close()
    samtools_view_process.wait()
    if tensor_file_path != "PIPE":
        tensor_fp.stdin.close()
        tensor_fp.wait()
        tensor_fpo.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate tensors summarizing local alignments from a BAM file and a list of candidate locations")

    parser.add_argument('--bam_fn', type=str, default="input.bam",
                        help="Sorted bam file input, default: %(default)s")

    parser.add_argument('--ref_fn', type=str, default="ref.fa",
                        help="Reference fasta file input, default: %(default)s")

    parser.add_argument('--can_fn', type=str, default="PIPE",
                        help="Variant candidate list generated by ExtractVariantCandidates.py or true variant list generated by GetTruth.py, use PIPE for standard input, default: %(default)s")

    parser.add_argument('--tensor_fn', type=str, default="PIPE",
                        help="Tensor output, use PIPE for standard output, default: %(default)s")

    parser.add_argument('--minMQ', type=int, default=0,
                        help="Minimum Mapping Quality. Mapping quality lower than the setting will be filtered, default: %(default)d")

    parser.add_argument('--ctgName', type=str, default="chr17",
                        help="The name of sequence to be processed, default: %(default)s")

    parser.add_argument('--ctgStart', type=int, default=None,
                        help="The 1-based starting position of the sequence to be processed")

    parser.add_argument('--ctgEnd', type=int, default=None,
                        help="The 1-based inclusive ending position of the sequence to be processed")

    parser.add_argument('--samtools', type=str, default="samtools",
                        help="Path to the 'samtools', default: %(default)s")

    parser.add_argument('--stop_consider_left_edge', action='store_true',
                        help="If not set, would consider left edge only. That is, count the left-most base-pairs of a read for coverage even if the starting position of a read is after the starting position of a tensor")

    parser.add_argument('--dcov', type=int, default=250,
                        help="Cap depth per position at %(default)d")

    parser.add_argument('--minCoverage', type=int, default=0,
                        help="Minimum coverage required to generate a tensor, default: %(default)d")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    OutputAlnTensor(args)


if __name__ == "__main__":
    main()
