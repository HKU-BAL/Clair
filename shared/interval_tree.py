import shlex
from intervaltree import IntervalTree

from shared.utils import subprocess_popen


def bed_tree_from(bed_file_path):
    """
    0-based interval tree [start, end)
    """

    tree = {}
    if bed_file_path is None:
        return tree

    unzip_process = subprocess_popen(shlex.split("gzip -fdc %s" % (bed_file_path)))
    while True:
        row = unzip_process.stdout.readline()
        is_finish_reading_output = row == '' and unzip_process.poll() is not None
        if is_finish_reading_output:
            break

        if row:
            columns = row.strip().split()

            ctg_name = columns[0]
            if ctg_name not in tree:
                tree[ctg_name] = IntervalTree()

            ctg_start, ctg_end = int(columns[1]), int(columns[2])
            if ctg_start == ctg_end:
                ctg_end += 1

            tree[ctg_name].addi(ctg_start, ctg_end)

    unzip_process.stdout.close()
    unzip_process.wait()

    return tree


def is_region_in(tree, contig_name, region_start=None, region_end=None):
    if (contig_name is None) or (contig_name not in tree):
        return False

    interval_tree = tree[contig_name]
    is_interval_tree_version_3 = hasattr(interval_tree, 'at')
    if is_interval_tree_version_3:
        return len(
            interval_tree.at(region_start)
            if region_end is None else
            interval_tree.overlap(begin=region_start, end=region_end)
        ) > 0

    # interval tree version 2
    return len(interval_tree.search(begin=region_start, end=region_end, strict=False)) > 0
