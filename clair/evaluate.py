import sys
import logging
import numpy as np
from os.path import abspath
from argparse import ArgumentParser
from time import time


from clair.model import Clair
import clair.utils as utils
from clair.task.main import GT21, GENOTYPE, VARIANT_LENGTH_1, VARIANT_LENGTH_2
import shared.param as param


logging.basicConfig(format='%(message)s', level=logging.INFO)


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
    dataset_size = dataset_info.dataset_size
    x_array_compressed = dataset_info.x_array_compressed
    y_array_compressed = dataset_info.y_array_compressed

    logging.info("[INFO] Testing on the training and validation dataset ...")
    prediction_start_time = time()
    prediction_batch_size = param.predictBatchSize

    no_of_training_examples = (
        dataset_info.no_of_training_examples_from_train_binary or int(dataset_size * param.trainingDatasetPercentage)
    )
    no_of_blosc_blocks = utils.no_of_blosc_blocks_from(
        dataset_info=dataset_info,
        no_of_training_examples=no_of_training_examples,
        blosc_block_size=param.bloscBlockSize
    )

    blosc_index = 0
    first_blosc_block_data_index = 0

    confusion_matrix_gt21 = new_confusion_matrix_with_dimension(GT21.output_label_count)
    confusion_matrix_genotype = new_confusion_matrix_with_dimension(GENOTYPE.output_label_count)
    confusion_matrix_indel_length_1 = new_confusion_matrix_with_dimension(VARIANT_LENGTH_1.output_label_count)
    confusion_matrix_indel_length_2 = new_confusion_matrix_with_dimension(VARIANT_LENGTH_2.output_label_count)

    all_gt21_count = top_1_count = top_2_count = 0

    while True:
        x_batch, next_x_first_blosc_block_data_index, next_x_blosc_index = utils.decompress_array(
            array=x_array_compressed,
            blosc_start_index=blosc_index,
            first_blosc_block_data_index=first_blosc_block_data_index,
            no_of_data_rows_to_retrieve=prediction_batch_size,
            no_of_blosc_blocks=no_of_blosc_blocks,
        )
        y_batch, _next_y_first_blosc_block_data_index, _next_y_blosc_index = utils.decompress_array(
            array=y_array_compressed,
            blosc_start_index=blosc_index,
            first_blosc_block_data_index=first_blosc_block_data_index,
            no_of_data_rows_to_retrieve=prediction_batch_size,
            no_of_blosc_blocks=no_of_blosc_blocks,
        )
        minibatch_gt21_prediction, minibatch_genotype_prediction, \
            minibatch_indel_length_prediction_1, minibatch_indel_length_prediction_2 = m.predict(x_batch)

        blosc_index = next_x_blosc_index
        first_blosc_block_data_index = next_x_first_blosc_block_data_index

        # update confusion matrix for gt21 prediction
        for gt21_prediction, gt21_label in zip(
            minibatch_gt21_prediction,
            y_batch[:, GT21.y_start_index:GT21.y_end_index]
        ):
            true_label_index = np.argmax(gt21_label)
            predict_label_index = np.argmax(gt21_prediction)
            confusion_matrix_gt21[true_label_index][predict_label_index] += 1

            all_gt21_count += 1
            indexes_with_sorted_prediction_probability = gt21_prediction.argsort()[::-1]
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

        if not (next_x_first_blosc_block_data_index >= 0 and next_x_blosc_index >= 0):
            break

    logging.info("[INFO] Prediciton time elapsed: %.2f s" % (time() - prediction_start_time))

    print("[INFO] Evaluation on gt21:")
    print("[INFO] all/top1/top2/top1p/top2p: %d/%d/%d/%.2f/%.2f" %
          (all_gt21_count, top_1_count, top_2_count,
           float(top_1_count)/all_gt21_count*100, float(top_2_count)/all_gt21_count*100))
    for i in range(GT21.output_label_count):
        print("\t".join([str(confusion_matrix_gt21[i][j]) for j in range(GT21.output_label_count)]))
    gt21_f_measure = f1_score(confusion_matrix_gt21)
    print("[INFO] f-measure: ", gt21_f_measure)

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


def main():
    parser = ArgumentParser(description="Evaluate trained model")

    parser.add_argument('--bin_fn', type=str, default=None,
                        help="Binary tensor input generated by tensor2Bin.py, tensor_fn, var_fn and bed_fn will be ignored")
    parser.add_argument('--train_bin_fn', type=str, default=None,
                        help="Train Binary, used together with --validation_bin_fn (would ignore: bin_fn, tensor_fn, var_fn, bed_fn)")
    parser.add_argument('--validation_bin_fn', type=str, default=None,
                        help="Validation Binary, used together with --train_bin_fn (would ignore: bin_fn, tensor_fn, var_fn, bed_fn)")

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

    m = Clair()
    m.init()

    dataset_info = utils.dataset_info_from(
        binary_file_path=args.bin_fn,
        tensor_file_path=args.tensor_fn,
        variant_file_path=args.var_fn,
        bed_file_path=args.bed_fn,
        train_binary_file_path=args.train_bin_fn,
        validation_binary_file_path=args.validation_bin_fn,
    )

    model_initalization_file_path = args.chkpnt_fn
    m.restore_parameters(abspath(model_initalization_file_path))

    # start evaluation
    evaluate_model(m, dataset_info)


if __name__ == "__main__":
    main()
