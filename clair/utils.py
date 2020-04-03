from __future__ import print_function

import sys
import gc
import shlex
import logging
import pickle
import numpy as np
import blosc
from os import environ
from enum import IntEnum
from collections import namedtuple

from clair.task.main import output_labels_from_reference, output_labels_from_vcf_columns
import shared.param as param
from shared.interval_tree import bed_tree_from, is_region_in
from shared.utils import subprocess_popen, IUPAC_base_to_num_dict as BASE2NUM, IUPAC_base_to_ACGT_base_dict as BASE2ACGT, BASIC_BASES

PREFIX_CHAR_STR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

DatasetInfo = namedtuple('DatasetInfo', [
    'dataset_size',
    'x_array_compressed',
    'y_array_compressed',
    'position_array_compressed',
    'no_of_training_examples_from_train_binary',
    'is_separated_train_and_validation_binary',
])
TrainingConfig = namedtuple('TrainingConfig', [
    'dataset_info',
    'learning_rate',
    'l2_regularization_lambda',
    'output_file_path_prefix',
    'model_initalization_file_path',
    'summary_writer'
])


def setup_environment():
    environ["CXX"] = "g++"
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    blosc.set_nthreads(4)
    gc.enable()


def blosc_pack_array(array):
    return blosc.pack_array(array, cname='lz4hc', clevel=9, shuffle=blosc.NOSHUFFLE)


def unpack_a_tensor_record(a, b, c, *d):
    return a, b, c, np.array(d, dtype=np.float32)


def batches_from(iterable, item_from, batch_size=1):
    iterable = iter(iterable)
    while True:
        chunk = []
        for _ in range(batch_size):
            try:
                chunk.append(item_from(next(iterable)))
            except StopIteration:
                yield chunk
                return
        yield chunk


no_of_positions, matrix_row, matrix_num = 2 * param.flankingBaseNum + 1, param.matrixRow, param.matrixNum
input_tensor_size = no_of_positions * matrix_row * matrix_num


def tensor_generator_from(tensor_file_path, batch_size):
    if tensor_file_path != "PIPE":
        f = subprocess_popen(shlex.split("gzip -fdc %s" % (tensor_file_path)))
        fo = f.stdout
    else:
        fo = sys.stdin

    processed_tensors = 0

    def item_from(row):
        columns = row.split()
        return (columns[:-input_tensor_size], np.array(columns[-input_tensor_size:], dtype=np.float32))

    for batch in batches_from(fo, item_from=item_from, batch_size=batch_size):
        tensors = np.empty((batch_size, input_tensor_size), dtype=np.float32)
        non_tensor_infos = []
        for non_tensor_info, tensor in batch:
            _, _, sequence = non_tensor_info
            if sequence[param.flankingBaseNum] not in BASE2NUM:
                continue
            tensors[len(non_tensor_infos)] = tensor
            non_tensor_infos.append(non_tensor_info)

        current_batch_size = len(non_tensor_infos)
        X = np.reshape(tensors, (batch_size, no_of_positions, matrix_row, matrix_num))
        for i in range(1, matrix_num):
            X[:current_batch_size, :, :, i] -= X[:current_batch_size, :, :, 0]

        processed_tensors += current_batch_size
        print("Processed %d tensors" % processed_tensors, file=sys.stderr)

        if current_batch_size <= 0:
            continue
        yield X[:current_batch_size], non_tensor_infos[:current_batch_size]

    if tensor_file_path != "PIPE":
        fo.close()
        f.wait()


def variant_map_from(var_fn, tree, is_tree_empty):
    Y = {}
    if var_fn is None:
        return Y

    f = subprocess_popen(shlex.split("gzip -fdc %s" % (var_fn)))
    for row in f.stdout:
        columns = row.split()
        ctg_name, position_str = columns[0], columns[1]

        if not (is_tree_empty or is_region_in(tree, ctg_name, int(position_str))):
            continue

        key = ctg_name + ":" + position_str
        Y[key] = output_labels_from_vcf_columns(columns)

    f.stdout.close()
    f.wait()
    return Y


def get_training_array(tensor_fn, var_fn, bed_fn, shuffle=True, is_allow_duplicate_chr_pos=False):
    tree = bed_tree_from(bed_file_path=bed_fn)
    is_tree_empty = len(tree.keys()) == 0

    Y = variant_map_from(var_fn, tree, is_tree_empty)

    X = {}
    f = subprocess_popen(shlex.split("gzip -fdc %s" % (tensor_fn)))
    total = 0
    mat = np.empty(input_tensor_size, dtype=np.float32)
    for row in f.stdout:
        chrom, coord, seq, mat = unpack_a_tensor_record(*(row.split()))
        if not (is_tree_empty or is_region_in(tree, chrom, int(coord))):
            continue
        seq = seq.upper()
        if seq[param.flankingBaseNum] not in BASIC_BASES:
            continue
        key = chrom + ":" + coord

        x = np.reshape(mat, (no_of_positions, matrix_row, matrix_num))
        for i in range(1, matrix_num):
            x[:, :, i] -= x[:, :, 0]

        if key not in X:
            X[key] = np.copy(x)
        elif is_allow_duplicate_chr_pos:
            new_key = ""
            for character in PREFIX_CHAR_STR:
                tmp_key = character + key
                if tmp_key not in X:
                    new_key = tmp_key
                    break
            if len(new_key) > 0:
                X[new_key] = np.copy(x)

        is_reference = key not in Y
        if is_reference:
            Y[key] = output_labels_from_reference(BASE2ACGT[seq[param.flankingBaseNum]])

        total += 1
        if total % 100000 == 0:
            print("Processed %d tensors" % total, file=sys.stderr)
    f.stdout.close()
    f.wait()

    # print "[INFO] size of X: {}, size of Y: {}".format(len(X), len(Y))

    all_chr_pos = sorted(X.keys())
    if shuffle == True:
        np.random.shuffle(all_chr_pos)

    X_compressed, Y_compressed, pos_compressed = [], [], []
    X_array, Y_array, pos_array = [], [], []
    count = 0
    total = 0
    for key in all_chr_pos:
        total += 1

        X_array.append(X[key])
        del X[key]

        if key in Y:
            Y_array.append(Y[key])
            pos_array.append(key)
            if not is_allow_duplicate_chr_pos:
                del Y[key]
        elif is_allow_duplicate_chr_pos:
            tmp_key = key[1:]
            Y_array.append(Y[tmp_key])
            pos_array.append(tmp_key)

        count += 1
        if count == param.bloscBlockSize:
            X_compressed.append(blosc_pack_array(np.array(X_array)))
            Y_compressed.append(blosc_pack_array(np.array(Y_array)))
            pos_compressed.append(blosc_pack_array(np.array(pos_array)))
            X_array, Y_array, pos_array = [], [], []
            count = 0

        if total % 50000 == 0:
            print("Compressed %d/%d tensor" % (total, len(all_chr_pos)), file=sys.stderr)

    if count > 0:
        X_compressed.append(blosc_pack_array(np.array(X_array)))
        Y_compressed.append(blosc_pack_array(np.array(Y_array)))
        pos_compressed.append(blosc_pack_array(np.array(pos_array)))

    return total, X_compressed, Y_compressed, pos_compressed


def decompress_array(
    array,
    blosc_start_index,
    first_blosc_block_data_index,
    no_of_data_rows_to_retrieve,
    no_of_blosc_blocks,
    read_index_list=None
):
    """
    Return:
        data_rows, next_first_blosc_block_data_index and next_blosc_start_index

    Note:
        blosc_start_index, next_first_blosc_block_data_index and next_blosc_start_index is inclusive.
    """
    data_rows = []
    no_of_data_rows = 0
    for i in range(blosc_start_index, no_of_blosc_blocks):
        new_data_rows = blosc.unpack_array(array[i if read_index_list is None else read_index_list[i]])
        data_rows.append(new_data_rows)
        no_of_data_rows += len(new_data_rows)

        if i == blosc_start_index and first_blosc_block_data_index > 0:
            return np.concatenate(data_rows[:])[first_blosc_block_data_index:], 0, i+1

        if no_of_data_rows >= no_of_data_rows_to_retrieve:
            extra_no_of_data_rows = no_of_data_rows % no_of_data_rows_to_retrieve
            next_blosc_start_index = i+1 if extra_no_of_data_rows == 0 else i
            next_first_blosc_block_data_index = (
                0 if extra_no_of_data_rows == 0 else (len(new_data_rows) - extra_no_of_data_rows)
            )
            return (
                np.concatenate(data_rows[:])[0:no_of_data_rows_to_retrieve],
                next_first_blosc_block_data_index if next_blosc_start_index < no_of_blosc_blocks else -1,
                next_blosc_start_index if next_blosc_start_index < no_of_blosc_blocks else -1
            )

    if no_of_data_rows <= 0:
        return None, -1, -1
    return np.concatenate(data_rows[:]), -1, -1


def dataset_info_from(
    binary_file_path,
    tensor_file_path=None,
    variant_file_path=None,
    bed_file_path=None,
    train_binary_file_path=None,
    validation_binary_file_path=None,
):
    logging.info("[INFO] Loading dataset...")
    no_of_training_examples_from_train_binary = None

    if train_binary_file_path is not None and validation_binary_file_path is not None:
        logging.info("[INFO] Loading compressed data from train and validation binary file path")
        with open(train_binary_file_path, "rb") as fh:
            dataset_size = pickle.load(fh)
            x_array_compressed = pickle.load(fh)
            y_array_compressed = pickle.load(fh)
            position_array_compressed = pickle.load(fh)
        no_of_training_examples_from_train_binary = dataset_size
        with open(validation_binary_file_path, "rb") as fh:
            dataset_size += pickle.load(fh)
            x_array_compressed += pickle.load(fh)
            y_array_compressed += pickle.load(fh)
            position_array_compressed += pickle.load(fh)

    elif binary_file_path != None:
        logging.info("[INFO] Loading compressed data from binary file path")
        with open(binary_file_path, "rb") as fh:
            dataset_size = pickle.load(fh)
            x_array_compressed = pickle.load(fh)
            y_array_compressed = pickle.load(fh)
            position_array_compressed = pickle.load(fh)
    else:
        logging.info("[INFO] Loading compressed data from utils get training array")
        dataset_size, x_array_compressed, y_array_compressed, position_array_compressed = \
            get_training_array(tensor_file_path, variant_file_path, bed_file_path)

    logging.info("[INFO] The size of dataset: {}".format(dataset_size))

    return DatasetInfo(
        dataset_size=dataset_size,
        x_array_compressed=x_array_compressed,
        y_array_compressed=y_array_compressed,
        position_array_compressed=position_array_compressed,
        no_of_training_examples_from_train_binary=no_of_training_examples_from_train_binary,
        is_separated_train_and_validation_binary=no_of_training_examples_from_train_binary is not None,
    )


def new_mini_batch(
    data_index,
    blosc_start_index,
    first_blosc_block_data_index,
    no_of_training_examples,
    no_of_blosc_blocks,
    dataset_info,
    tensor_block_index_list
):
    """
    Return:
        x_batch, y_batch, next_first_blosc_block_data_index, next_blosc_index
    """
    if blosc_start_index >= no_of_blosc_blocks:
        return None, None, -1, -1

    x_array_compressed = dataset_info.x_array_compressed
    y_array_compressed = dataset_info.y_array_compressed
    training_batch_size = param.trainBatchSize
    validation_batch_size = param.predictBatchSize
    is_training = data_index < no_of_training_examples
    is_validation = not is_training

    # calculate new batch size according to dataset index
    # train: 0 - validation_data_start_index - 1, validation: validation_data_start_index - dataset_size
    if is_training and (no_of_training_examples - data_index) < training_batch_size:
        batch_size = no_of_training_examples - data_index
    elif is_training:
        batch_size = training_batch_size
    elif is_validation:
        batch_size = validation_batch_size

    def decompress_array_from(array):
        return decompress_array(
            array=array,
            blosc_start_index=blosc_start_index,
            first_blosc_block_data_index=first_blosc_block_data_index,
            no_of_data_rows_to_retrieve=batch_size,
            no_of_blosc_blocks=no_of_blosc_blocks,
            read_index_list=tensor_block_index_list
        )
    x_batch, next_x_first_blosc_block_data_index, next_x_blosc_index = decompress_array_from(x_array_compressed)
    y_batch, _next_y_first_blosc_block_data_index, next_y_blosc_index = decompress_array_from(y_array_compressed)

    x_batch_size, y_batch_size = np.shape(x_batch)[0], np.shape(y_batch)[0]
    x_end_flag, y_end_flag = next_x_blosc_index == -1, next_y_blosc_index == -1
    if x_batch_size != y_batch_size or x_end_flag != y_end_flag:
        sys.exit("[ERROR] Inconsistency between decompressed arrays: %d/%d" % (x_batch_size, y_batch_size))

    return x_batch, y_batch, next_x_first_blosc_block_data_index, next_x_blosc_index


def no_of_blosc_blocks_from(
    dataset_info,
    no_of_training_examples,
    blosc_block_size,
):
    if dataset_info.is_separated_train_and_validation_binary:
        no_of_validation_examples = dataset_info.dataset_size - no_of_training_examples
        no_of_training_blocks = int(np.ceil(float(no_of_training_examples) / blosc_block_size))
        no_of_validation_blocks = int(np.ceil(float(no_of_validation_examples) / blosc_block_size))
        return no_of_training_blocks + no_of_validation_blocks

    return int(np.ceil(float(dataset_info.dataset_size) / blosc_block_size))
