from os.path import isfile, abspath
from sys import exit, stderr
from subprocess import check_output, PIPE, Popen

# A->A
# C->C
# G->G
# T or U->T
# R->A or G
# Y->C or T
# S->G or C
# W->A or T
# K->G or T
# M->A or C
# B->C or G or T
# D->A or G or T
# H->A or C or T
# V->A or C or G
IUPAC_base_to_ACGT_base_dict = dict(zip(
    "ACGTURYSWKMBDHVN",
    ("A", "C", "G", "T", "T", "A", "C", "C", "A", "G", "A", "C", "A", "A", "A", "A")
))

IUPAC_base_to_num_dict = dict(zip(
    "ACGTURYSWKMBDHVN",
    (0, 1, 2, 3, 3, 0, 1, 1, 0, 2, 0, 1, 0, 0, 0, 0)
))

BASIC_BASES = set("ACGTU")

def is_file_exists(file_name, suffix=""):
    if not isinstance(file_name, str) or not isinstance(suffix, str):
        return False
    return isfile(file_name + suffix)


def file_path_from(file_name, suffix="", exit_on_not_found=False):
    if is_file_exists(file_name, suffix):
        return abspath(file_name)
    if exit_on_not_found:
        exit("[ERROR] file %s not found" % (file_name + suffix))
    return None


def is_command_exists(command):
    if not isinstance(command, str):
        return False

    try:
        check_output("which %s" % (command), shell=True)
        return True
    except:
        return False


def executable_command_string_from(command_to_execute, exit_on_not_found=False):
    if is_command_exists(command_to_execute):
        return command_to_execute
    if exit_on_not_found:
        exit("[ERROR] %s executable not found" % (command_to_execute))
    return None


def subprocess_popen(args, stdin=None, stdout=PIPE, stderr=stderr, bufsize=8388608):
    return Popen(args, stdin=stdin, stdout=stdout, stderr=stderr, bufsize=bufsize, universal_newlines=True)
