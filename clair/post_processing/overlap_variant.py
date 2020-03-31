from sys import stdin, stderr
from collections import namedtuple

Variant = namedtuple('Variant', [
    'chromosome',
    'position',
    'reference_base',
    'alternate_base',
    'alternate_base_multi',
    'quality_score',
    'genotype',
    'depth',
    'allele_frequency',
])

VariantIntervals = namedtuple('VariantIntervals', [
    'snp_interval',
    'deletion_interval',
    'insertion_intervals',
])


EMPTY_INTERVAL = (-1, -1)


DEBUG_OVERLAPPED_VARIANT = False


def maximum_deletion_length_of(variant):
    return len(variant.reference_base) - min(
        len(variant.alternate_base),
        1024 if variant.alternate_base_multi is None else len(variant.alternate_base_multi),
    )


def snp_interval_from(variant):
    # need to handle the case like [ACGT]Del / [ACGT]Ins
    is_snp = (
        len(variant.reference_base) == len(variant.alternate_base) or
        (
            False if variant.alternate_base_multi is None else len(
                variant.reference_base) == len(variant.alternate_base_multi)
        )
    )
    return EMPTY_INTERVAL if not is_snp else (variant.position - 1, variant.position)


def deletion_interval_from(variant):
    maximum_deletion_length = maximum_deletion_length_of(variant)
    is_deletion = maximum_deletion_length > 0

    return EMPTY_INTERVAL if not is_deletion else (variant.position - 1, variant.position + maximum_deletion_length)


def insertion_intervals_from(variant):
    insertion_intervals = []

    if len(variant.alternate_base) > len(variant.reference_base):
        insertion_intervals.append(
            (
                variant.position - 1,
                variant.position + len(variant.alternate_base) - len(variant.reference_base)
            )
        )
    else:
        insertion_intervals.append(EMPTY_INTERVAL)

    if (
        variant.alternate_base_multi is not None and
        len(variant.alternate_base_multi) > len(variant.reference_base)
    ):
        insertion_intervals.append(
            (
                variant.position - 1,
                variant.position + len(variant.alternate_base_multi) - len(variant.reference_base)
            )
        )
    else:
        insertion_intervals.append(EMPTY_INTERVAL)

    return insertion_intervals


# all intervals is suppose to be zero-base and [start, end) half open interval
def variant_intervals_from(variant):
    return VariantIntervals(
        snp_interval=snp_interval_from(variant),
        deletion_interval=deletion_interval_from(variant),
        insertion_intervals=insertion_intervals_from(variant),
    )


def is_two_intervals_overlap(interval1, interval2):
    if interval1 is EMPTY_INTERVAL or interval2 is EMPTY_INTERVAL:
        return False

    begin1, end1 = interval1
    begin2, _ = interval2
    # return begin1 <= begin2 <= end1 or begin2 <= end1 <= end2
    return begin1 <= begin2 < end1


def is_two_intervals_overlap_for_ins_snp(insertion_interval, snp_interval):
    if insertion_interval is EMPTY_INTERVAL or snp_interval is EMPTY_INTERVAL:
        return False

    insert_begin, insert_end = insertion_interval
    _, snp_end = snp_interval
    return insert_end - insert_begin == 2 and insert_end == snp_end


# for insertion intervals overlap, current implementation needs with the same ending position
def is_two_intervals_overlap_for_ins_ins(interval1, interval2):
    if interval1 is EMPTY_INTERVAL or interval2 is EMPTY_INTERVAL:
        return False

    _, end1 = interval1
    _, end2 = interval2
    return end1 == end2


def is_two_variants_overlap(variant1, variant2):
    if variant1.chromosome != variant2.chromosome:
        return False
    if variant1.position > variant2.position:
        return is_two_variants_overlap(variant2, variant1)

    intervals_1 = variant_intervals_from(variant1)
    intervals_2 = variant_intervals_from(variant2)

    # return (
    #     is_two_intervals_overlap(intervals_1.deletion_interval, intervals_2.snp_interval) or
    #     is_two_intervals_overlap(intervals_1.deletion_interval, intervals_2.deletion_interval) or
    #     is_two_intervals_overlap_for_ins_snp(intervals_1.insertion_intervals[0], intervals_2.snp_interval) or
    #     is_two_intervals_overlap_for_ins_snp(intervals_1.insertion_intervals[1], intervals_2.snp_interval) or
    #     is_two_intervals_overlap_for_ins_ins(intervals_1.insertion_intervals[0], intervals_2.insertion_intervals[0]) or
    #     is_two_intervals_overlap_for_ins_ins(intervals_1.insertion_intervals[0], intervals_2.insertion_intervals[1]) or
    #     is_two_intervals_overlap_for_ins_ins(intervals_1.insertion_intervals[1], intervals_2.insertion_intervals[0]) or
    #     is_two_intervals_overlap_for_ins_ins(intervals_1.insertion_intervals[1], intervals_2.insertion_intervals[1])
    # )

    # return (
    #     is_two_intervals_overlap(intervals_1.deletion_interval, intervals_2.snp_interval) or
    #     is_two_intervals_overlap(intervals_1.deletion_interval, intervals_2.deletion_interval) or
    #     is_two_intervals_overlap_for_ins_snp(intervals_1.insertion_intervals[0], intervals_2.snp_interval) or
    #     is_two_intervals_overlap_for_ins_snp(intervals_1.insertion_intervals[1], intervals_2.snp_interval)
    # )

    return (
        is_two_intervals_overlap(intervals_1.deletion_interval, intervals_2.snp_interval) or
        is_two_intervals_overlap(intervals_1.deletion_interval, intervals_2.deletion_interval)
    )


def variant_from(variant_row):
    if variant_row[0] == "#":
        return

    columns = str(variant_row).split("\t")
    chromosome = columns[0]
    position = int(columns[1])

    reference_base = columns[3]
    alternates = columns[4].split(",")
    alternate_base = alternates[0]
    alternate_base_multi = None if len(alternates) == 1 else alternates[1]

    quality_score = int(float(columns[5]))

    last_column = columns[-1]
    last_columns = last_column.split(":")
    genotype = last_columns[0]
    depth = last_columns[2]
    allele_frequency = last_columns[3]

    return Variant(
        chromosome=chromosome,
        position=position,
        reference_base=reference_base,
        alternate_base=alternate_base,
        alternate_base_multi=alternate_base_multi,
        quality_score=quality_score,
        genotype=genotype,
        depth=depth,
        allele_frequency=allele_frequency,
    )


def variant_row_from(variant):
    alternates = ",".join(
        [variant.alternate_base] +
        ([] if variant.alternate_base_multi is None else [variant.alternate_base_multi])
    )
    quality_score_str = str(variant.quality_score)
    last_column = ":".join([
        variant.genotype,
        quality_score_str,
        variant.depth,
        variant.allele_frequency,
    ])

    return "\t".join([
        variant.chromosome,
        str(variant.position),
        ".",
        variant.reference_base,
        alternates,
        str(variant.quality_score),
        ".",
        ".",
        "GT:GQ:DP:AF",
        last_column,
    ])


def header_and_variant_rows_from_stdin():
    header_rows = []
    variant_rows = []
    for row in stdin.readlines():
        if row[0] == "#":
            header_rows.append(row[:-1])
        else:
            variant_rows.append(row[:-1])

    return header_rows, variant_rows


def variant_to_output_for(variant1, variant2):
    # return variant1 if variant1.quality_score > variant2.quality_score else variant2
    score1 = variant1.quality_score
    score2 = variant2.quality_score
    # score1 = variant1.quality_score * float(variant1.allele_frequency)
    # score2 = variant2.quality_score * float(variant2.allele_frequency)
    return variant1 if score1 > score2 else variant2


def filter_variants_with(variants):
    filtered_variants = []

    overlapped_variants_count = 0

    for variant in variants:
        if len(filtered_variants) == 0:
            filtered_variants.append(variant)
            continue

        last_variant = filtered_variants[-1]
        if not is_two_variants_overlap(last_variant, variant):
            filtered_variants.append(variant)
            continue

        if DEBUG_OVERLAPPED_VARIANT:
            overlapped_variants_count += 1
            print("\n[INFO] variants overlapped.", file=stderr)
            print(variant_row_from(last_variant), file=stderr)
            print(variant_row_from(variant), file=stderr)

        # variant_to_append = last_variant if last_variant.quality_score >= variant.quality_score else variant
        variant_to_append = variant_to_output_for(last_variant, variant)
        if variant_to_append != last_variant:
            filtered_variants.pop()
            filtered_variants.append(variant)

    if DEBUG_OVERLAPPED_VARIANT:
        print("[INFO] {} variants overlapped.".format(overlapped_variants_count), file=stderr)

    return filtered_variants


def output(header_rows, variants):
    for header_row in header_rows:
        print(header_row)
    for variant in variants:
        print(variant_row_from(variant))


def main():
    header_rows, variant_rows = header_and_variant_rows_from_stdin()
    variants = [variant_from(variant_row) for variant_row in variant_rows]
    filtered_variants = filter_variants_with(variants)
    output(header_rows, filtered_variants)


if __name__ == "__main__":
    main()
