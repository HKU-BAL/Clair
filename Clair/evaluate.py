import sys
import os
import time
import argparse
import logging
import numpy as np

import param
import utils
import clair_model as cv
from utils import BASE_CHANGE, GENOTYPE, VARIANT_LENGTH_1, VARIANT_LENGTH_2

logging.basicConfig(format='%(message)s', level=logging.INFO)
base2num = dict(zip("ACGT", (0, 1, 2, 3)))
num2base = dict(zip((0, 1, 2, 3), "ACGT"))


def f1_score(confusion_matrix):
    column_sum = confusion_matrix.sum(axis=0)
    row_sum = confusion_matrix.sum(axis=1)

    f1_score_array = np.array([])
    matrix_size = confusion_matrix.shape[0]
    epsilon = 1e-15
    for i in range(matrix_size):
        TP = confusion_matrix[i][i] + 0.0
        precision = TP / (column_sum[i] + epsilon)
        recall = TP / (row_sum[i] + epsilon)
        f1_score_array = np.append(f1_score_array, (2.0 * precision * recall) / (precision + recall + epsilon))

    return f1_score_array


def new_confusion_matrix_with_dimension(size):
    return np.zeros((size, size), dtype=np.int)


def evaluate_model(m, dataset_info):
    dataset_size = dataset_info["dataset_size"]
    x_array_compressed = dataset_info["x_array_compressed"]
    y_array_compressed = dataset_info["y_array_compressed"]

    logging.info("[INFO] Testing on the training and validation dataset ...")
    prediction_start_time = time.time()
    prediction_batch_size = param.predictBatchSize
    # no_of_training_examples = int(dataset_size*param.trainingDatasetPercentage)
    # validation_data_start_index = no_of_training_examples + 1

    dataset_index = 0
    x_end_flag = 0
    y_end_flag = 0

    confusion_matrix_base = new_confusion_matrix_with_dimension(BASE_CHANGE.output_label_count)
    confusion_matrix_genotype = new_confusion_matrix_with_dimension(GENOTYPE.output_label_count)
    confusion_matrix_indel_length_1 = new_confusion_matrix_with_dimension(VARIANT_LENGTH_1.output_label_count)
    confusion_matrix_indel_length_2 = new_confusion_matrix_with_dimension(VARIANT_LENGTH_2.output_label_count)

    all_base_count = top_1_count = top_2_count = 0

    while dataset_index < dataset_size:
        if x_end_flag != 0 or y_end_flag != 0:
            break

        x_batch, _, x_end_flag = utils.decompress_array(
            x_array_compressed, dataset_index, prediction_batch_size, dataset_size)
        y_batch, _, y_end_flag = utils.decompress_array(
            y_array_compressed, dataset_index, prediction_batch_size, dataset_size)
        minibatch_base_prediction, minibatch_genotype_prediction, \
            minibatch_indel_length_prediction_1, minibatch_indel_length_prediction_2 = m.predict(x_batch)
        dataset_index += prediction_batch_size

        # update confusion matrix for base prediction
        for base_change_prediction, base_change_label in zip(
            minibatch_base_prediction,
            y_batch[:, BASE_CHANGE.y_start_index:BASE_CHANGE.y_end_index]
        ):
            true_label_index = np.argmax(base_change_label)
            predict_label_index = np.argmax(base_change_prediction)
            confusion_matrix_base[true_label_index][predict_label_index] += 1

            all_base_count += 1
            indexes_with_sorted_prediction_probability = base_change_prediction.argsort()[::-1]
            if true_label_index == indexes_with_sorted_prediction_probability[0]:
                top_1_count += 1
                top_2_count += 1
            elif true_label_index == indexes_with_sorted_prediction_probability[1]:
                top_2_count += 1

        # update confusion matrix for genotype
        for genotype_prediction, true_genotype_label in zip(
            minibatch_genotype_prediction,
            y_batch[:, GENOTYPE.y_start_index:GENOTYPE.y_end_index]
        ):
            confusion_matrix_genotype[np.argmax(true_genotype_label)][np.argmax(genotype_prediction)] += 1

        # update confusion matrix for indel length 1 and 2
        for indel_length_prediction_1, true_indel_length_label_1, indel_length_prediction_2, true_indel_length_label_2 in zip(
            minibatch_indel_length_prediction_1,
            y_batch[:, VARIANT_LENGTH_1.y_start_index:VARIANT_LENGTH_1.y_end_index],
            minibatch_indel_length_prediction_2,
            y_batch[:, VARIANT_LENGTH_2.y_start_index:VARIANT_LENGTH_2.y_end_index]
        ):
            true_label_index_1 = np.argmax(true_indel_length_label_1)
            true_label_index_2 = np.argmax(true_indel_length_label_2)
            predict_label_index_1 = np.argmax(indel_length_prediction_1)
            predict_label_index_2 = np.argmax(indel_length_prediction_2)

            if true_label_index_1 > true_label_index_2:
                true_label_index_1, true_label_index_2 = true_label_index_2, true_label_index_1
            if predict_label_index_1 > predict_label_index_2:
                predict_label_index_1, predict_label_index_2 = predict_label_index_2, predict_label_index_1

            confusion_matrix_indel_length_1[true_label_index_1][predict_label_index_1] += 1
            confusion_matrix_indel_length_2[true_label_index_2][predict_label_index_2] += 1

    logging.info("[INFO] Prediciton time elapsed: %.2f s" % (time.time() - prediction_start_time))

    print("[INFO] Evaluation on base change:")
    print("[INFO] all/top1/top2/top1p/top2p: %d/%d/%d/%.2f/%.2f" %
          (all_base_count, top_1_count, top_2_count,
           float(top_1_count)/all_base_count*100, float(top_2_count)/all_base_count*100))
    for i in range(BASE_CHANGE.output_label_count):
        print("\t".join([str(confusion_matrix_base[i][j]) for j in range(BASE_CHANGE.output_label_count)]))
    base_change_f_measure = f1_score(confusion_matrix_base)
    print("[INFO] f-measure: ", base_change_f_measure)

    print("\n[INFO] Evaluation on Genotype:")
    for i in range(GENOTYPE.output_label_count):
        print("\t".join([str(confusion_matrix_genotype[i][j]) for j in range(GENOTYPE.output_label_count)]))
    genotype_f_measure = f1_score(confusion_matrix_genotype)
    print("[INFO] f-measure: ", genotype_f_measure)

    print("\n[INFO] evaluation on indel length 1:")
    for i in range(VARIANT_LENGTH_1.output_label_count):
        print("\t".join([str(confusion_matrix_indel_length_1[i][j])
                         for j in range(VARIANT_LENGTH_1.output_label_count)]))
    indel_length_f_measure_1 = f1_score(confusion_matrix_indel_length_1)
    print("[INFO] f-measure: ", indel_length_f_measure_1)

    print("\n[INFO] evaluation on indel length 2:")
    for i in range(VARIANT_LENGTH_2.output_label_count):
        print("\t".join([str(confusion_matrix_indel_length_2[i][j])
                         for j in range(VARIANT_LENGTH_2.output_label_count)]))
    indel_length_f_measure_2 = f1_score(confusion_matrix_indel_length_2)
    print("[INFO] f-measure: ", indel_length_f_measure_2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate trained Clair model")

    parser.add_argument('--bin_fn', type=str, default=None,
                        help="Binary tensor input generated by tensor2Bin.py, tensor_fn, var_fn and bed_fn will be ignored")

    parser.add_argument('--tensor_fn', type=str, default="vartensors",
                        help="Tensor input")

    parser.add_argument('--var_fn', type=str, default="truthvars",
                        help="Truth variants list input")

    parser.add_argument('--bed_fn', type=str, default=None,
                        help="High confident genome regions input in the BED format")

    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a checkpoint for testing, REQUIRED")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    # initialize
    logging.info("[INFO] Loading model ...")
    utils.setup_environment()

    m = cv.Clair()
    m.init()

    dataset_info = utils.dataset_info_from(
        binary_file_path=args.bin_fn,
        tensor_file_path=args.tensor_fn,
        variant_file_path=args.var_fn,
        bed_file_path=args.bed_fn
    )

    model_initalization_file_path = args.chkpnt_fn
    m.restore_parameters(os.path.abspath(model_initalization_file_path))

    # start evaluation
    evaluate_model(m, dataset_info)
