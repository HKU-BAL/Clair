from __future__ import print_function

import sys
import os
import logging
import blosc
import numpy as np
from argparse import ArgumentParser
from threading import Thread

import clair.utils as utils
import shared.param as param

logging.basicConfig(format='%(message)s', level=logging.INFO)


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
        x_batch, y_batch, pos_batch, next_first_blosc_block_data_index, next_blosc_index
    """
    if blosc_start_index >= no_of_blosc_blocks:
        return None, None, -1, -1

    x_array_compressed = dataset_info.x_array_compressed
    y_array_compressed = dataset_info.y_array_compressed
    position_array_compressed = dataset_info.position_array_compressed
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
        return utils.decompress_array(
            array=array,
            blosc_start_index=blosc_start_index,
            first_blosc_block_data_index=first_blosc_block_data_index,
            no_of_data_rows_to_retrieve=batch_size,
            no_of_blosc_blocks=no_of_blosc_blocks,
            read_index_list=tensor_block_index_list
        )
    x_batch, next_x_first_blosc_block_data_index, next_x_blosc_index = decompress_array_from(x_array_compressed)
    y_batch, _next_y_first_blosc_block_data_index, next_y_blosc_index = decompress_array_from(y_array_compressed)
    pos_batch, _, _ = decompress_array_from(position_array_compressed)

    x_batch_size, y_batch_size = np.shape(x_batch)[0], np.shape(y_batch)[0]
    x_end_flag, y_end_flag = next_x_blosc_index == -1, next_y_blosc_index == -1
    if x_batch_size != y_batch_size or x_end_flag != y_end_flag:
        sys.exit("[ERROR] Inconsistency between decompressed arrays: %d/%d" % (x_batch_size, y_batch_size))

    return x_batch, y_batch, pos_batch, next_x_first_blosc_block_data_index, next_x_blosc_index
    # return x_batch, y_batch, None, next_x_first_blosc_block_data_index, next_x_blosc_index


def load_model(dataset_info):
    dataset_size = dataset_info.dataset_size

    # Model Constants
    no_of_blosc_blocks = utils.no_of_blosc_blocks_from(
        dataset_info=dataset_info,
        no_of_training_examples=dataset_size,
        blosc_block_size=param.bloscBlockSize
    )
    # tensor_block_index_list = np.arange(no_of_blosc_blocks, dtype=int)
    tensor_block_index_list = None

    # Initialize variables
    data_index = 0
    blosc_index = 0
    first_blosc_block_data_index = 0
    x_batch = None
    y_batch = None
    pos_batch = None

    while True:
        is_with_batch_data = x_batch is not None and y_batch is not None

        thread_pool = []
        if is_with_batch_data:
            for x_tensor, y_tensor, pos in zip(x_batch, y_batch, pos_batch):
                x_array = ["%d" % int(x_float) for x_float in list(x_tensor.flatten())]
                y_array = ["%d" % y_number for y_number in list(y_tensor.flatten())]

                print(" ".join(x_array))    # print x_array
                print(" ".join(y_array))    # print y_array
                print(pos)                  # print pos
        for t in thread_pool:
            t.start()

        try:
            # next_x_batch, next_y_batch, _next_pos_batch, next_first_blosc_block_data_index, next_blosc_start_index = new_mini_batch(
            next_x_batch, next_y_batch, next_pos_batch, next_first_blosc_block_data_index, next_blosc_start_index = new_mini_batch(
                data_index=data_index,
                blosc_start_index=blosc_index,
                first_blosc_block_data_index=first_blosc_block_data_index,
                no_of_training_examples=dataset_size,
                no_of_blosc_blocks=no_of_blosc_blocks,
                dataset_info=dataset_info,
                tensor_block_index_list=tensor_block_index_list,
            )
        except:
            print("Error catched")
            print(next_first_blosc_block_data_index, next_blosc_start_index)

        # wait until loaded next mini batch & finished training/validation with current mini batch
        for t in thread_pool:
            t.join()

        if next_x_batch is not None:
            batch_size = np.shape(next_x_batch)[0]
            data_index += batch_size
            blosc_index = next_blosc_start_index
            first_blosc_block_data_index = next_first_blosc_block_data_index

        # if not go through whole dataset yet, continue the process
        if next_first_blosc_block_data_index >= 0 and next_blosc_start_index >= 0:
            x_batch = next_x_batch
            y_batch = next_y_batch
            pos_batch = next_pos_batch
            continue

        # break after loaded all data from bin
        break


def export_model(binary_file_path):
    try:
        import cPickle as pickle
    except:
        import pickle

    def pickle_dump(obj, file):
        return pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

    def blosc_pack_array(array):
        return blosc.pack_array(array, cname='lz4hc', clevel=9, shuffle=blosc.NOSHUFFLE)

    row_index = 0
    x_compressed = []
    y_compressed = []
    pos_compressed = []

    total = 0
    x = []
    y = []
    pos = []

    for line in sys.stdin:
        is_x = row_index % 3 == 0
        is_y = row_index % 3 == 1
        is_pos = row_index % 3 == 2

        if is_x:
            total += 1
            x.append(list(map(int, line.split(" "))))
        elif is_y:
            y.append(list(map(int, line.split(" "))))
        elif is_pos:
            pos.append(line)

        row_index += 1
        row_index %= 3

        if total % param.bloscBlockSize == 0 and row_index == 0:
            x_compressed.append(blosc_pack_array(np.array(x).reshape(-1, 33, 8, 4)))
            y_compressed.append(blosc_pack_array(np.array(y)))
            pos_compressed.append(blosc_pack_array(np.array(pos)))
            x, y, pos = [], [], []

            if total % 5000 == 0:
                logging.error("[INFO] Processed %d tensors" % total)

    if len(x) > 0:
        x_compressed.append(blosc_pack_array(np.array(x).reshape(-1, 33, 8, 4)))
        y_compressed.append(blosc_pack_array(np.array(y)))
        pos_compressed.append(blosc_pack_array(np.array(pos)))
        x, y, pos = [], [], []
        logging.error("[INFO] Processed %d tensors" % total)

    logging.error("[INFO] Writing to binary ...")
    with open(binary_file_path, 'wb') as f:
        pickle_dump(total, f)
        pickle_dump(x_compressed, f)
        pickle_dump(y_compressed, f)
        pickle_dump(pos_compressed, f)


def main():
    parser = ArgumentParser(description="Load bin using python2, export bin using python3")

    parser.add_argument('--is_export', action='store_true',
                        help="If this option enabled, it is used to export instead of import")

    parser.add_argument('--bin_fn', type=str, default=None,
                        help="If --is_export enabled, this is the export bin file path. If --is_export not enabled, this is the import bin file path")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    # initialize
    logging.error("[INFO] Initializing")

    if args.is_export:
        export_model(binary_file_path=args.bin_fn)
    else:
        utils.setup_environment()
        load_model(utils.dataset_info_from(binary_file_path=args.bin_fn))


if __name__ == "__main__":
    main()
