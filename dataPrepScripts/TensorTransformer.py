import numpy as np
from sys import stdin

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
need_ref_seq_update = source_no_of_flanking_bases != target_no_of_flanking_bases

# A, C, G, T
no_of_base_labels = 4
# reference, ins, del, snp
no_of_channels = 4

source_size_of_strand_channel = no_of_base_labels * (2 if is_source_tensor_have_strand_infomation else 1)
source_tensor_size = (source_no_of_flanking_bases * 2 + 1) * source_size_of_strand_channel * no_of_channels

target_size_of_strand_channel = no_of_base_labels * (2 if is_target_tensor_have_strand_information else 1)
target_tensor_size = (target_no_of_flanking_bases * 2 + 1) * target_size_of_strand_channel * no_of_channels


difference_in_number_of_flanking_bases = source_no_of_flanking_bases - target_no_of_flanking_bases
# target_tensor_start_index is inclusive, target_tensor_end_index is exclusive
target_tensor_start_index = difference_in_number_of_flanking_bases * source_size_of_strand_channel * no_of_channels
target_tensor_end_index =  source_tensor_size - target_tensor_start_index

# ref_seq_start_index is inclusive, ref_seq_end_index is exclusive
ref_seq_start_index = source_no_of_flanking_bases - target_no_of_flanking_bases
ref_seq_end_index = source_no_of_flanking_bases + target_no_of_flanking_bases + 1

for row in stdin:
    columns = row.strip().split()
    ctg_name, ctg_pos, ref_seq, tensor = columns[0], columns[1], columns[2], columns[3:]

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

    if need_ref_seq_update:
        ref_seq = ref_seq[ref_seq_start_index:ref_seq_end_index]

    print("{} {} {} {}".format(ctg_name, ctg_pos, ref_seq, " ".join("%.0f" % x for x in tensor)))
