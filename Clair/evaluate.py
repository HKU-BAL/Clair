import sys
import os
import time
import argparse
import logging
import pickle
import numpy as np

import param
import utils
import clair as cv

logging.basicConfig(format='%(message)s', level=logging.INFO)
base2num = dict(zip("ACGT", (0, 1, 2, 3)))
num2base = dict(zip((0, 1, 2, 3), "ACGT"))

def f1_score(confusion_matrix):
    column_sum = confusion_matrix.sum(axis=0)
    row_sum = confusion_matrix.sum(axis=1)

    f1_score_array = np.array([])
    matrix_size = confusion_matrix.shape[0]
    for i in range(matrix_size):
        TP = confusion_matrix[i][i] + 0.0
        precision = TP / column_sum[i]
        recall = TP / row_sum[i]
        f1_score_array = np.append(f1_score_array, (2.0 * precision * recall) / (precision + recall))

    return f1_score_array


def evaluate_model(m, dataset_info):
    dataset_size = dataset_info["dataset_size"]
    x_array_compressed = dataset_info["x_array_compressed"]
    y_array_compressed = dataset_info["y_array_compressed"]

    logging.info("[INFO] Testing on the training and validation dataset ...")
    prediction_start_time = time.time()
    prediction_batch_size = param.predictBatchSize
    # no_of_training_examples = int(dataset_size*param.trainingDatasetPercentage)
    # validation_data_start_index = no_of_training_examples + 1

    base_predictions = []
    zygosity_predictions = []
    variant_type_predictions = []
    indel_length_predictions = []

    dataset_index = 0
    end_flag = 0

    while dataset_index < dataset_size:
        if end_flag != 0:
            break

        x_batch, _, end_flag = utils.decompress_array(
            x_array_compressed, dataset_index, prediction_batch_size, dataset_size)
        minibatch_base_prediction, minibatch_zygosity_prediction, \
            minibatch_variant_type_prediction, minibatch_indel_length_prediction = m.predict(x_batch)

        base_predictions.append(minibatch_base_prediction)
        zygosity_predictions.append(minibatch_zygosity_prediction)
        variant_type_predictions.append(minibatch_variant_type_prediction)
        indel_length_predictions.append(minibatch_indel_length_prediction)

        dataset_index += prediction_batch_size

    base_predictions = np.concatenate(base_predictions[:])
    zygosity_predictions = np.concatenate(zygosity_predictions[:])
    variant_type_predictions = np.concatenate(variant_type_predictions[:])
    indel_length_predictions = np.concatenate(indel_length_predictions[:])

    logging.info("[INFO] Prediciton time elapsed: %.2f s" % (time.time() - prediction_start_time))

    # Evaluate the trained model
    y_array, _, _ = utils.decompress_array(y_array_compressed, 0, dataset_size, dataset_size)

    logging.info("[INFO] Evaluation on base change:")

    print("[INFO] Evaluation on base change:")
    all_base_count = top_1_count = top_2_count = 0
    confusion_matrix = np.zeros((4, 4), dtype=np.int)
    for base_change_prediction, base_change_label in zip(base_predictions, y_array[:, 0:4]):
        confusion_matrix[np.argmax(base_change_label)][np.argmax(base_change_prediction)] += 1

        all_base_count += 1
        indexes_with_sorted_prediction_probability = base_change_prediction.argsort()[::-1]
        if np.argmax(base_change_label) == indexes_with_sorted_prediction_probability[0]:
            top_1_count += 1
            top_2_count += 1
        elif np.argmax(base_change_label) == indexes_with_sorted_prediction_probability[1]:
            top_2_count += 1

    print("[INFO] all/top1/top2/top1p/top2p: %d/%d/%d/%.2f/%.2f" %
                 (all_base_count, top_1_count, top_2_count,
                  float(top_1_count)/all_base_count*100, float(top_2_count)/all_base_count*100))
    for i in range(4):
        print("\t".join([str(confusion_matrix[i][j]) for j in range(4)]))
    base_change_f_measure = f1_score(confusion_matrix)
    print("[INFO] f-measure: ", base_change_f_measure)

    # Zygosity
    print("\n[INFO] Evaluation on Zygosity:")
    confusion_matrix = np.zeros((2, 2), dtype=np.int)
    for zygosity_prediction, true_zygosity_label in zip(zygosity_predictions, y_array[:, 4:6]):
        confusion_matrix[np.argmax(true_zygosity_label)][np.argmax(zygosity_prediction)] += 1
    for epoch_count in range(2):
        print("\t".join([str(confusion_matrix[epoch_count][j]) for j in range(2)]))
    zygosity_f_measure = f1_score(confusion_matrix)
    print("[INFO] f-measure: ", zygosity_f_measure)

    # Variant type
    print("\n[INFO] Evaluation on variant type:")
    confusion_matrix = np.zeros((4, 4), dtype=np.int)
    for variant_type_prediction, true_variant_type_label in zip(variant_type_predictions, y_array[:, 6:10]):
        confusion_matrix[np.argmax(true_variant_type_label)][np.argmax(variant_type_prediction)] += 1
    for i in range(4):
        print("\t".join([str(confusion_matrix[i][j]) for j in range(4)]))
    variant_type_f_measure = f1_score(confusion_matrix)
    print("[INFO] f-measure: ", variant_type_f_measure)

    # Indel length
    print("\n[INFO] evaluation on indel length:")
    confusion_matrix = np.zeros((6, 6), dtype=np.int)
    for indel_length_prediction, true_indel_length_label in zip(indel_length_predictions, y_array[:, 10:16]):
        confusion_matrix[np.argmax(true_indel_length_label)][np.argmax(indel_length_prediction)] += 1
    for i in range(6):
        print("\t".join([str(confusion_matrix[i][j]) for j in range(6)]))
    indel_length_f_measure = f1_score(confusion_matrix)[:-1]
    print("[INFO] f-measure: ", indel_length_f_measure)

    print("[INFO] base change f-measure mean: %.6f" % np.mean(base_change_f_measure))
    print("[INFO] zygosity f-measure mean: %.6f" % np.mean(zygosity_f_measure))
    print("[INFO] variant type f-measure mean: %.6f" % np.mean(variant_type_f_measure))
    print("[INFO] indel length f-measure mean: %.6f" % np.mean(indel_length_f_measure))
    print("[INFO] f-measure mean: %.6f" % np.mean([
        np.mean(base_change_f_measure),
        np.mean(zygosity_f_measure),
        np.mean(variant_type_f_measure),
        np.mean(indel_length_f_measure)
    ]))


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
