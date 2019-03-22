import os
import sys
import gc
import shlex
import subprocess
import logging
import pickle
import numpy as np

import intervaltree
import blosc
import param
from enum import IntEnum

base2num = dict(zip("ACGT", (0, 1, 2, 3)))
PREFIX_CHAR_STR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class BaseChange(IntEnum):
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
        'AA': BaseChange.AA,
        'AC': BaseChange.AC,
        'AG': BaseChange.AG,
        'AT': BaseChange.AT,
        'CC': BaseChange.CC,
        'CG': BaseChange.CG,
        'CT': BaseChange.CT,
        'GG': BaseChange.GG,
        'GT': BaseChange.GT,
        'TT': BaseChange.TT,
        'DelDel': BaseChange.DelDel,
        'ADel': BaseChange.ADel,
        'CDel': BaseChange.CDel,
        'GDel': BaseChange.GDel,
        'TDel': BaseChange.TDel,
        'InsIns': BaseChange.InsIns,
        'AIns': BaseChange.AIns,
        'CIns': BaseChange.CIns,
        'GIns': BaseChange.GIns,
        'TIns': BaseChange.TIns,
        'InsDel': BaseChange.InsDel,
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
    return base_change_label_from(BaseChange.InsDel)


class Genotype(IntEnum):
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


def SetupEnv():
    os.environ["CXX"] = "g++"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    blosc.set_nthreads(2)
    gc.enable()


def UnpackATensorRecord(a, b, c, *d):
    return a, b, c, np.array(d, dtype=np.float32)


def GetTensor(tensor_fn, num):
    if tensor_fn != "PIPE":
        f = subprocess.Popen(shlex.split("gzip -fdc %s" % (tensor_fn)), stdout=subprocess.PIPE, bufsize=8388608)
        fo = f.stdout
    else:
        fo = sys.stdin
    total = 0
    c = 0
    rows = np.empty((num, ((2*param.flankingBaseNum+1)*param.matrixRow*param.matrixNum)), dtype=np.float32)
    pos = []
    for row in fo:  # A variant per row
        try:
            chrom, coord, seq, rows[c] = UnpackATensorRecord(*(row.split()))
        except ValueError:
            print >> sys.stderr, "UnpackATensorRecord Failure", row
        seq = seq.upper()
        if seq[param.flankingBaseNum] not in ["A", "C", "G", "T"]:  # TODO: Support IUPAC in the future
            continue
        pos.append(chrom + ":" + coord + ":" + seq)
        c += 1

        if c == num:
            x = np.reshape(rows, (num, 2*param.flankingBaseNum+1, param.matrixRow, param.matrixNum))

            for i in range(1, param.matrixNum):
                x[:, :, :, i] -= x[:, :, :, 0]
            total += c
            print >> sys.stderr, "Processed %d tensors" % total
            yield 0, c, x, pos
            c = 0
            rows = np.empty((num, ((2*param.flankingBaseNum+1)*param.matrixRow*param.matrixNum)), dtype=np.float32)
            pos = []

    if tensor_fn != "PIPE":
        fo.close()
        f.wait()
    x = np.reshape(rows[:c], (c, 2*param.flankingBaseNum+1, param.matrixRow, param.matrixNum))
    for i in range(1, param.matrixNum):
        x[:, :, :, i] -= x[:, :, :, 0]
    total += c
    print >> sys.stderr, "Processed %d tensors" % total
    yield 1, c, x, pos


def GetTrainingArray(tensor_fn, var_fn, bed_fn, shuffle=True, is_allow_duplicate_chr_pos=False):
    tree = {}
    if bed_fn != None:
        f = subprocess.Popen(shlex.split("gzip -fdc %s" % (bed_fn)), stdout=subprocess.PIPE, bufsize=8388608)
        for row in f.stdout:
            row = row.split()
            name = row[0]
            if name not in tree:
                tree[name] = intervaltree.IntervalTree()
            begin = int(row[1])
            end = int(row[2])-1
            if end == begin:
                end += 1
            tree[name].addi(begin, end)
        f.stdout.close()
        f.wait()

    Y = {}
    if var_fn != None:
        f = subprocess.Popen(shlex.split("gzip -fdc %s" % (var_fn)), stdout=subprocess.PIPE, bufsize=8388608)
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
            #                  AA  AC  AG  AT  CC  CG  CT  GG  GT  TT  DD  AD  CD  GD  TD  II  AI  CI  GI  TI  ID
            base_change_vec = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            partial_labels = [partial_label_from(reference, alternate) for alternate in alternate_arr]
            base_change_label = mix_two_partial_labels(partial_labels[0], partial_labels[1])
            base_change = base_change_enum_from(base_change_label)
            base_change_vec[base_change] = 1

            # geno type
            #               0/0 1/1 0/1
            genotype_vec = [0., 0., 0.]
            is_homo_reference = genotype_1 == "0" and genotype_2 == "0"
            is_homo_variant = not is_homo_reference and genotype_1 == genotype_2
            is_hetero_variant = not is_homo_reference and not is_homo_variant
            is_multi = not is_homo_variant and genotype_1 != "0" and genotype_2 != "0"
            if is_homo_reference:
                genotype_vec[Genotype.homo_reference] = 1.0
            elif is_homo_variant:
                genotype_vec[Genotype.homo_variant] = 1.0
            elif is_hetero_variant and not is_multi:
                genotype_vec[Genotype.hetero_variant] = 1.0
            elif is_hetero_variant and is_multi:
                genotype_vec[Genotype.hetero_variant] = 1.0
                # genotype_vec[Genotype.hetero_variant_multi] = 1.0

            # variant length
            variant_length_vec = [
                max(
                    min(len(alternate) - len(reference), param.flankingBaseNum),
                    -param.flankingBaseNum
                ) for alternate in alternate_arr
            ]
            variant_length_vec.sort()

            Y[key] = base_change_vec + genotype_vec + variant_length_vec

        f.stdout.close()
        f.wait()

    X = {}
    f = subprocess.Popen(shlex.split("gzip -fdc %s" % (tensor_fn)), stdout=subprocess.PIPE, bufsize=8388608)
    total = 0
    mat = np.empty(((2*param.flankingBaseNum+1)*param.matrixRow*param.matrixNum), dtype=np.float32)
    for row in f.stdout:
        chrom, coord, seq, mat = UnpackATensorRecord(*(row.split()))
        if bed_fn != None:
            if chrom not in tree:
                continue
            if len(tree[chrom].search(int(coord))) == 0:
                continue
        seq = seq.upper()
        if seq[param.flankingBaseNum] not in ["A", "C", "G", "T"]:
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
            #                  AA  AC  AG  AT  CC  CG  CT  GG  GT  TT  DD  AD  CD  GD  TD  II  AI  CI  GI  TI  ID
            base_change_vec = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            #               0/0 1/1 0/1
            genotype_vec = [1., 0., 0.]
            #                     L1  L2
            variant_length_vec = [0., 0.]

            base_change_vec[base_change_enum_from(seq[param.flankingBaseNum] + seq[param.flankingBaseNum])] = 1

            Y[key] = base_change_vec + genotype_vec + variant_length_vec

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
            XArrayCompressed.append(blosc.pack_array(np.array(XArray), cname='lz4hc'))
            YArrayCompressed.append(blosc.pack_array(np.array(YArray), cname='lz4hc'))
            posArrayCompressed.append(blosc.pack_array(np.array(posArray), cname='lz4hc'))
            XArray = []
            YArray = []
            posArray = []
            count = 0
        if total % 50000 == 0:
            print >> sys.stderr, "Compressed %d/%d tensor" % (total, len(allPos))
    if count > 0:
        XArrayCompressed.append(blosc.pack_array(np.array(XArray), cname='lz4hc'))
        YArrayCompressed.append(blosc.pack_array(np.array(YArray), cname='lz4hc'))
        posArrayCompressed.append(blosc.pack_array(np.array(posArray), cname='lz4hc'))

    return total, XArrayCompressed, YArrayCompressed, posArrayCompressed


def DecompressArray(array, start, num, maximum):
    endFlag = 0
    if start + num >= maximum:
        num = maximum - start
        endFlag = 1
    leftEnd = start % param.bloscBlockSize
    startingBlock = int(start / param.bloscBlockSize)
    maximumBlock = int((start+num-1) / param.bloscBlockSize)
    rt = []
    rt.append(blosc.unpack_array(array[startingBlock]))
    startingBlock += 1
    if startingBlock <= maximumBlock:
        for i in range(startingBlock, (maximumBlock+1)):
            rt.append(blosc.unpack_array(array[i]))
    nprt = np.concatenate(rt[:])
    if leftEnd != 0 or num % param.bloscBlockSize != 0:
        nprt = nprt[leftEnd:(leftEnd+num)]

    return nprt, num, endFlag


def DecompressArray_with_order(array, start, num, maximum, read_index_list=None):
    endFlag = 0
    if start + num >= maximum:
        num = maximum - start
        endFlag = 1
    leftEnd = start % param.bloscBlockSize
    startingBlock = int(start / param.bloscBlockSize)
    maximumBlock = int((start+num-1) / param.bloscBlockSize)
    rt = []
    rt.append(blosc.unpack_array(array[startingBlock]))
    startingBlock += 1
    if startingBlock <= maximumBlock:
        if read_index_list is None:
            for i in range(startingBlock, (maximumBlock+1)):
                rt.append(blosc.unpack_array(array[i]))
        else:
            for i in range(startingBlock, (maximumBlock+1)):
                rt.append(blosc.unpack_array(array[read_index_list[i]]))
    nprt = np.concatenate(rt[:])
    if leftEnd != 0 or num % param.bloscBlockSize != 0:
        nprt = nprt[leftEnd:(leftEnd+num)]

    return nprt, num, endFlag


def dataset_info_from(binary_file_path, tensor_file_path=None, variant_file_path=None, bed_file_path=None):
    logging.info("[INFO] Loading dataset...")

    if binary_file_path != None:
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

    return dict(
        dataset_size=dataset_size,
        x_array_compressed=x_array_compressed,
        y_array_compressed=y_array_compressed,
        position_array_compressed=position_array_compressed
    )


# function aliases
def setup_environment():
    return SetupEnv()


def unpack_a_tensor_record(a, b, c, *d):
    return a, b, c, np.array(d, dtype=np.float32)


def get_tensor(tensor_fn, num):
    return GetTensor(tensor_fn, num)


def get_training_array(tensor_fn, var_fn, bed_fn, shuffle=True, is_allow_duplicate_chr_pos=False):
    return GetTrainingArray(tensor_fn, var_fn, bed_fn, shuffle, is_allow_duplicate_chr_pos)


def decompress_array(array, start, num, maximum):
    return DecompressArray(array, start, num, maximum)


def decompress_array_with_order(array, start, num, maximum, read_index_list=None):
    return DecompressArray_with_order(array, start, num, maximum, read_index_list)
