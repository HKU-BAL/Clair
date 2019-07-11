import sys
import argparse
import subprocess
import shlex
import os
import re
import gc
import numpy as np

source_no_of_flanking_bases = 32
target_no_of_flanking_bases = 16
if source_no_of_flanking_bases < target_no_of_flanking_bases:
    print("[ERROR] Impossible to transform a tensor with source # of flanking bases < target # of flanking bases.")
    exit(1)


is_source_tensor_have_strand_infomation = True
is_target_tensor_have_strand_information = True
if is_source_tensor_have_strand_infomation is False and is_target_tensor_have_strand_information is True:
    print("[ERROR] Impossible to transform a tensor without strand information to a tensor with strand information.")
    exit(1)

need_remove_strand_information = (
    is_source_tensor_have_strand_infomation is True and is_target_tensor_have_strand_information is False
)


# A, C, G, T
no_of_base_labels = 4
# reference, ins, del, snp
no_of_channels = 4

source_size_of_strand_channel = no_of_base_labels * (2 if is_source_tensor_have_strand_infomation else 1)
source_tensor_size = (source_no_of_flanking_bases * 2 + 1) * source_size_of_strand_channel * no_of_channels

target_size_of_strand_channel = no_of_base_labels * (2 if is_target_tensor_have_strand_information else 1)
target_tensor_size = (target_no_of_flanking_bases * 2 + 1) * target_size_of_strand_channel * no_of_channels


difference_in_number_of_flanking_bases = source_no_of_flanking_bases - target_no_of_flanking_bases
# target_tensor_start_index is inclusive, while target_tensor_end_index is exclusive
target_tensor_start_index = difference_in_number_of_flanking_bases * source_size_of_strand_channel * no_of_channels
target_tensor_end_index =  source_tensor_size - target_tensor_start_index

for row in sys.stdin:
    columns = row.strip().split()
    ctg_name, center, ref_seq, tensor = columns[0], columns[1], columns[2], columns[3:]

    if len(tensor) != source_tensor_size:
        print("[ERROR] Unexpected tensor size. Expected: {}, given: {}.".format(source_tensor_size, len(tensor)))
        exit(1)

    tensor = tensor[target_tensor_start_index:target_tensor_end_index]
    tensor = [float(x) for x in tensor]

    if need_remove_strand_information:
        tensor = np.array(tensor)
        tensor = tensor.reshape(2 * target_no_of_flanking_bases + 1, source_size_of_strand_channel, no_of_channels)

        tensor[:,:no_of_base_labels] += tensor[:,no_of_base_labels:]
        tensor = tensor[:,:no_of_base_labels]

        tensor = tensor.flatten()

    # print("%s %s %s %s" % (ctg_name, center, ref_seq, " ".join("%0.1f" % float(x) for x in tensor)))
    print("{} {} {} {}".format(ctg_name, center, ref_seq, " ".join("%.0f" % x for x in tensor)))
