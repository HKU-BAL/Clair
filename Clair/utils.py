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

base2num = dict(zip("ACGT", (0, 1, 2, 3)))
PREFIX_CHAR_STR = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


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
            ctgName = row[0]
            pos = int(row[1])
            if bed_fn != None:
                if len(tree[ctgName].search(pos)) == 0:
                    continue
            key = ctgName + ":" + str(pos)

            baseVec = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            #          --------------  ------  ------------    ------------------
            #          Base chng       Zygo.   Var type        Var length
            #          A   C   G   T   HET HOM REF SNP INS DEL 0   1   2   3   4   >=4
            #          0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15

            if row[4] == "0" and row[5] == "1":
                if len(row[2]) == 1 and len(row[3]) == 1:
                    baseVec[base2num[row[2][0]]] = 0.5
                    baseVec[base2num[row[3][0]]] = 0.5
                elif len(row[2]) > 1 or len(row[3]) > 1:
                    baseVec[base2num[row[2][0]]] = 0.5
                baseVec[4] = 1.

            elif row[4] == "1" and row[5] == "1":
                if len(row[2]) == 1 and len(row[3]) == 1:
                    baseVec[base2num[row[3][0]]] = 1
                elif len(row[2]) > 1 or len(row[3]) > 1:
                    pass
                baseVec[5] = 1.

            if len(row[2]) > 1 and len(row[3]) == 1:
                baseVec[9] = 1.  # deletion
            elif len(row[3]) > 1 and len(row[2]) == 1:
                baseVec[8] = 1.  # insertion
            else:
                baseVec[7] = 1.  # SNP

            varLen = abs(len(row[2])-len(row[3]))
            if varLen > 4:
                baseVec[15] = 1.
            else:
                baseVec[10+varLen] = 1.

            Y[key] = baseVec
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

        if key not in Y:
            baseVec = [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
            #          --------------  ------  ------------    ------------------
            #          Base chng       Zygo.   Var type        Var length
            #          A   C   G   T   HET HOM REF SNP INS DEL 0   1   2   3   4   >=4
            #          0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15

            baseVec[base2num[seq[param.flankingBaseNum]]] = 1.
            Y[key] = baseVec

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
