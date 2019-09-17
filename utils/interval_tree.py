import subprocess
import shlex
import intervaltree

def interval_tree_from(bed_file_path):
    """
    0-based interval tree [start, end)
    """

    tree = {}
    if bed_file_path is None:
        return tree

    unzip_process = subprocess.Popen(
        shlex.split("pigz -fdc %s" % (bed_file_path)),
        stdout=subprocess.PIPE,
        bufsize=8388608
    )
    while True:
        row = unzip_process.stdout.readline()
        is_finish_reading_output = row == '' and unzip_process.poll() is not None
        if is_finish_reading_output:
            break

        if row:
            columns = row.strip().split()

            ctg_name = columns[0]
            if ctg_name not in tree:
                tree[ctg_name] = intervaltree.IntervalTree()

            ctg_start, ctg_end = int(columns[1]), int(columns[2])
            if ctg_start == ctg_end:
                ctg_end += 1

            tree[ctg_name].addi(ctg_start, ctg_end)

    unzip_process.stdout.close()
    unzip_process.wait()

    return tree
