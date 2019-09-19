from os.path import isfile, abspath
from sys import exit
from subprocess import check_output


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
