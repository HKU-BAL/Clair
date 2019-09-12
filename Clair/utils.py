import os
import sys
import gc
import shlex
import subprocess
import logging
import cPickle
import numpy as np

import intervaltree
import blosc
import param
from enum import IntEnum

from collections import namedtuple

BASES = set("ACGT")
base2num = dict(zip("ACGT", (0, 1, 2, 3)))
PREFIX_CHAR_STR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

VariantLengthNamedTuple = namedtuple('VariantLengthNamedTuple', ['index_offset', 'min', 'max', 'output_label_count'])
variant_length_index_offset = 16
VariantLength = VariantLengthNamedTuple(
    index_offset=variant_length_index_offset,
    min=-variant_length_index_offset,
    max=variant_length_index_offset,
    output_label_count=variant_length_index_offset * 2 + 1,
)

OutputLabelNamedTuple = namedtuple('BasePredictNamedTuple', ['output_label_count', 'y_start_index', 'y_end_index'])
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

BASE_CHANGE = OutputLabelNamedTuple(
    output_label_count=21,
    y_start_index=0,
    y_end_index=21,
)
GENOTYPE = OutputLabelNamedTuple(
    output_label_count=3,
    y_start_index=BASE_CHANGE.y_end_index,
    y_end_index=BASE_CHANGE.y_end_index + 3,
)
VARIANT_LENGTH_1 = OutputLabelNamedTuple(
    output_label_count=VariantLength.output_label_count,
    y_start_index=GENOTYPE.y_end_index,
    y_end_index=GENOTYPE.y_end_index + VariantLength.output_label_count,
)
VARIANT_LENGTH_2 = OutputLabelNamedTuple(
    output_label_count=VariantLength.output_label_count,
    y_start_index=VARIANT_LENGTH_1.y_end_index,
    y_end_index=VARIANT_LENGTH_1.y_end_index + VariantLength.output_label_count,
)


class GT21(IntEnum):
    AA = 0
    AC = 1
    AG = 2
    AT = 3
    CC = 4
    CG = 5
    CT = 6
    GG = 7
    GT = 8
    TT = 9
    DelDel = 10
    ADel = 11
    CDel = 12
    GDel = 13
    TDel = 14
    InsIns = 15
    AIns = 16
    CIns = 17
    GIns = 18
    TIns = 19
    InsDel = 20


def base_change_label_from(base_change_enum):
    return [
        'AA',
        'AC',
        'AG',
        'AT',
        'CC',
        'CG',
        'CT',
        'GG',
        'GT',
        'TT',
        'DelDel',
        'ADel',
        'CDel',
        'GDel',
        'TDel',
        'InsIns',
        'AIns',
        'CIns',
        'GIns',
        'TIns',
        'InsDel'
    ][base_change_enum]


def base_change_enum_from(base_change_label):
    return {
        'AA': GT21.AA,
        'AC': GT21.AC,
        'AG': GT21.AG,
        'AT': GT21.AT,
        'CC': GT21.CC,
        'CG': GT21.CG,
        'CT': GT21.CT,
        'GG': GT21.GG,
        'GT': GT21.GT,
        'TT': GT21.TT,
        'DelDel': GT21.DelDel,
        'ADel': GT21.ADel,
        'CDel': GT21.CDel,
        'GDel': GT21.GDel,
        'TDel': GT21.TDel,
        'InsIns': GT21.InsIns,
        'AIns': GT21.AIns,
        'CIns': GT21.CIns,
        'GIns': GT21.GIns,
        'TIns': GT21.TIns,
        'InsDel': GT21.InsDel,
    }[base_change_label]


def partial_label_from(ref, alt):
    if len(ref) > len(alt):
        return "Del"
    elif len(ref) < len(alt):
        return "Ins"
    return alt[0]


def mix_two_partial_labels(label1, label2):
    # AA, AC, AG, AT, CC, CG, CT, GG, GT, TT
    if len(label1) == 1 and len(label2) == 1:
        return label1 + label2 if label1 <= label2 else label2 + label1

    # ADel, CDel, GDel, TDel, AIns, CIns, GIns, TIns
    tmp_label1, tmp_label2 = label1, label2
    if len(label1) > 1 and len(label2) == 1:
        tmp_label1, tmp_label2 = label2, label1
    if len(tmp_label2) > 1 and len(tmp_label1) == 1:
        return tmp_label1 + tmp_label2

    # InsIns, DelDel
    if len(label1) > 0 and len(label2) > 0 and label1 == label2:
        return label1 + label2

    # InsDel
    return base_change_label_from(GT21.InsDel)


class Genotype(IntEnum):
    unknown = -1
    homo_reference = 0          # 0/0
    homo_variant = 1            # 1/1
    hetero_variant = 2          # 0/1 OR 1/2
    hetero_variant_multi = 3    # 1/2


def genotype_string_from(genotype):
    if genotype == Genotype.homo_reference:
        return "0/0"
    elif genotype == Genotype.homo_variant:
        return "1/1"
    elif genotype == Genotype.hetero_variant:
        return "0/1"
    elif genotype == Genotype.hetero_variant_multi:
        return "1/2"
    return ""


def setup_environment():
    os.environ["CXX"] = "g++"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
        for _ in xrange(batch_size):
            try:
                chunk.append(item_from(next(iterable)))
            except StopIteration:
                yield chunk
                return
        yield chunk


def tensor_generator_from(tensor_file_path, batch_size):
    if tensor_file_path != "PIPE":
        f = subprocess.Popen(shlex.split("pigz -fdc %s" % (tensor_file_path)), stdout=subprocess.PIPE, bufsize=8388608)
        fo = f.stdout
    else:
        fo = sys.stdin

    processed_tensors = 0

    no_of_positions, matrix_row, matrix_num = 2 * param.flankingBaseNum + 1, param.matrixRow, param.matrixNum
    input_tensor_size = no_of_positions * matrix_row * matrix_num

    def item_from(row):
        columns = row.split()
        return (columns[:-input_tensor_size], np.array(columns[-input_tensor_size:], dtype=np.float32))

    for batch in batches_from(fo, item_from=item_from, batch_size=batch_size):
        tensors = np.empty((batch_size, input_tensor_size), dtype=np.float32)
        non_tensor_infos = []
        for non_tensor_info, tensor in batch:
            _, _, sequence = non_tensor_info
            if sequence[param.flankingBaseNum] not in BASES:  # TODO: Support IUPAC in the future
                continue
            tensors[len(non_tensor_infos)] = tensor
            non_tensor_infos.append(non_tensor_info)

        current_batch_size = len(non_tensor_infos)
        X = np.reshape(tensors, (batch_size, no_of_positions, matrix_row, matrix_num))
        for i in range(1, matrix_num):
            X[:current_batch_size, :, :, i] -= X[:current_batch_size, :, :, 0]

        processed_tensors += current_batch_size
        print >> sys.stderr, "Processed %d tensors" % processed_tensors

        if current_batch_size <= 0:
            continue
        yield X[:current_batch_size], non_tensor_infos[:current_batch_size]

    if tensor_file_path != "PIPE":
        fo.close()
        f.wait()


def get_training_array(tensor_fn, var_fn, bed_fn, shuffle=True, is_allow_duplicate_chr_pos=False):
    tree = {}
    if bed_fn != None:
        f = subprocess.Popen(shlex.split("pigz -fdc %s" % (bed_fn)), stdout=subprocess.PIPE, bufsize=8388608)
        for row in f.stdout:
            row = row.split()
            name = row[0]
            if name not in tree:
                tree[name] = intervaltree.IntervalTree()
            begin = int(row[1])
            end = int(row[2])
            if end == begin:
                end += 1
            tree[name].addi(begin, end)
        f.stdout.close()
        f.wait()

    Y = {}
    if var_fn != None:
        f = subprocess.Popen(shlex.split("pigz -fdc %s" % (var_fn)), stdout=subprocess.PIPE, bufsize=8388608)
        for row in f.stdout:
            row = row.split()
            ctg_name = row[0]
            position_str = row[1]

            if bed_fn != None:
                if len(tree[ctg_name].search(int(position_str))) == 0:
                    continue
            key = ctg_name + ":" + position_str

            reference = row[2]
            alternate_arr = row[3].split(',')
            genotype_1, genotype_2 = row[4], row[5]
            if int(genotype_1) > int(genotype_2):
                genotype_1, genotype_2 = genotype_2, genotype_1
            if len(alternate_arr) == 1:
                alternate_arr = (
                    [reference if genotype_1 == "0" or genotype_2 == "0" else alternate_arr[0]] +
                    alternate_arr
                )

            # base change
            base_change_vec = [0] * BASE_CHANGE.output_label_count
            partial_labels = [partial_label_from(reference, alternate) for alternate in alternate_arr]
            base_change_label = mix_two_partial_labels(partial_labels[0], partial_labels[1])
            base_change = base_change_enum_from(base_change_label)
            base_change_vec[base_change] = 1

            # geno type
            genotype_vec = [0] * GENOTYPE.output_label_count
            is_homo_reference = genotype_1 == "0" and genotype_2 == "0"
            is_homo_variant = not is_homo_reference and genotype_1 == genotype_2
            is_hetero_variant = not is_homo_reference and not is_homo_variant
            is_multi = not is_homo_variant and genotype_1 != "0" and genotype_2 != "0"
            if is_homo_reference:
                genotype_vec[Genotype.homo_reference] = 1
            elif is_homo_variant:
                genotype_vec[Genotype.homo_variant] = 1
            elif is_hetero_variant and not is_multi:
                genotype_vec[Genotype.hetero_variant] = 1
            elif is_hetero_variant and is_multi:
                genotype_vec[Genotype.hetero_variant] = 1
                # genotype_vec[Genotype.hetero_variant_multi] = 1

            # variant length
            variant_lengths = [max(
                min(len(alternate) - len(reference), VariantLength.max),
                VariantLength.min
            ) for alternate in alternate_arr]
            variant_lengths.sort()
            variant_length_vec_1 = [0] * VARIANT_LENGTH_1.output_label_count
            variant_length_vec_2 = [0] * VARIANT_LENGTH_2.output_label_count
            variant_length_vec_1[variant_lengths[0] + VariantLength.index_offset] = 1
            variant_length_vec_2[variant_lengths[1] + VariantLength.index_offset] = 1

            Y[key] = base_change_vec + genotype_vec + variant_length_vec_1 + variant_length_vec_2

        f.stdout.close()
        f.wait()

    X = {}
    f = subprocess.Popen(shlex.split("pigz -fdc %s" % (tensor_fn)), stdout=subprocess.PIPE, bufsize=8388608)
    total = 0
    mat = np.empty(((2*param.flankingBaseNum+1)*param.matrixRow*param.matrixNum), dtype=np.float32)
    for row in f.stdout:
        chrom, coord, seq, mat = unpack_a_tensor_record(*(row.split()))
        if bed_fn != None:
            if chrom not in tree:
                continue
            if len(tree[chrom].search(int(coord))) == 0:
                continue
        seq = seq.upper()
        if seq[param.flankingBaseNum] not in BASES:
            continue
        key = chrom + ":" + coord

        x = np.reshape(mat, (2*param.flankingBaseNum+1, param.matrixRow, param.matrixNum))
        for i in range(1, param.matrixNum):
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
            base_change_vec = [0] * BASE_CHANGE.output_label_count
            base_change_vec[base_change_enum_from(seq[param.flankingBaseNum] + seq[param.flankingBaseNum])] = 1

            genotype_vec = [0] * GENOTYPE.output_label_count
            genotype_vec[Genotype.homo_reference] = 1

            variant_length_vec_1 = [0] * VARIANT_LENGTH_1.output_label_count
            variant_length_vec_2 = [0] * VARIANT_LENGTH_2.output_label_count
            variant_length_vec_1[0 + VariantLength.index_offset] = 1
            variant_length_vec_2[0 + VariantLength.index_offset] = 1

            Y[key] = base_change_vec + genotype_vec + variant_length_vec_1 + variant_length_vec_2

        total += 1
        if total % 100000 == 0:
            print >> sys.stderr, "Processed %d tensors" % total
    f.stdout.close()
    f.wait()

    # print "[INFO] size of X: {}, size of Y: {}".format(len(X), len(Y))

    allPos = sorted(X.keys())
    if shuffle == True:
        np.random.shuffle(allPos)

    XArrayCompressed = []
    YArrayCompressed = []
    posArrayCompressed = []
    XArray = []
    YArray = []
    posArray = []
    count = 0
    total = 0
    for key in allPos:
        total += 1

        XArray.append(X[key])
        del X[key]

        if key in Y:
            YArray.append(Y[key])
            posArray.append(key)
            if not is_allow_duplicate_chr_pos:
                del Y[key]
        elif is_allow_duplicate_chr_pos:
            tmp_key = key[1:]
            YArray.append(Y[tmp_key])
            posArray.append(tmp_key)

        count += 1
        if count == param.bloscBlockSize:
            XArrayCompressed.append(blosc_pack_array(np.array(XArray)))
            YArrayCompressed.append(blosc_pack_array(np.array(YArray)))
            posArrayCompressed.append(blosc_pack_array(np.array(posArray)))
            XArray = []
            YArray = []
            posArray = []
            count = 0
        if total % 50000 == 0:
            print >> sys.stderr, "Compressed %d/%d tensor" % (total, len(allPos))
    if count > 0:
        XArrayCompressed.append(blosc_pack_array(np.array(XArray)))
        YArrayCompressed.append(blosc_pack_array(np.array(YArray)))
        posArrayCompressed.append(blosc_pack_array(np.array(posArray)))

    return total, XArrayCompressed, YArrayCompressed, posArrayCompressed


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
    for i in xrange(blosc_start_index, no_of_blosc_blocks):
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
                next_first_blosc_block_data_index,
                next_blosc_start_index
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
            dataset_size = cPickle.load(fh)
            x_array_compressed = cPickle.load(fh)
            y_array_compressed = cPickle.load(fh)
            position_array_compressed = cPickle.load(fh)
        no_of_training_examples_from_train_binary = dataset_size
        with open(validation_binary_file_path, "rb") as fh:
            dataset_size += cPickle.load(fh)
            x_array_compressed += cPickle.load(fh)
            y_array_compressed += cPickle.load(fh)
            position_array_compressed += cPickle.load(fh)

    elif binary_file_path != None:
        logging.info("[INFO] Loading compressed data from binary file path")
        with open(binary_file_path, "rb") as fh:
            dataset_size = cPickle.load(fh)
            x_array_compressed = cPickle.load(fh)
            y_array_compressed = cPickle.load(fh)
            position_array_compressed = cPickle.load(fh)
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
    y_batch, _next_x_first_blosc_block_data_index, next_y_blosc_index = decompress_array_from(y_array_compressed)

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
