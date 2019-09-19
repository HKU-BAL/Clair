import sys
import os
import time
import argparse
import logging
import random

import numpy as np
import pandas as pd
from threading import Thread

import utils
import evaluate
import clair_model as cv
import shared.param as param
from utils import BASE_CHANGE, GENOTYPE, VARIANT_LENGTH_1, VARIANT_LENGTH_2

logging.basicConfig(format='%(message)s', level=logging.INFO)


def accuracy(base, genotype, indel_length_1, indel_length_2, y_batch):
    base_samples = len(base)+0.0
    genotype_samples = len(genotype)+0.0
    indel1_samples = len(indel_length_1)+0.0
    indel2_samples = len(indel_length_2)+0.0
    base_TP = 0.0
    genotype_TP = 0.0
    indel1_TP = 0.0
    indel2_TP = 0.0

    for base_change_prediction, base_change_label in zip(
            base,
            y_batch[:, BASE_CHANGE.y_start_index:BASE_CHANGE.y_end_index]
    ):
        true_label_index = np.argmax(base_change_label)
        predict_label_index = np.argmax(base_change_prediction)
        if true_label_index == predict_label_index:
            base_TP += 1

    for genotype_prediction, true_genotype_label in zip(
            genotype,
            y_batch[:, GENOTYPE.y_start_index:GENOTYPE.y_end_index]
    ):
        true_label_index = np.argmax(true_genotype_label)
        predict_label_index = np.argmax(genotype_prediction)
        if true_label_index == predict_label_index:
            genotype_TP += 1

    for indel_length_prediction_1, true_indel_length_label_1, indel_length_prediction_2, true_indel_length_label_2 in zip(
        indel_length_1,
        y_batch[:, VARIANT_LENGTH_1.y_start_index:VARIANT_LENGTH_1.y_end_index],
        indel_length_2,
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

        if true_label_index_1 == predict_label_index_1:
            indel1_TP += 1
        if true_label_index_2 == predict_label_index_2:
            indel2_TP += 1

    base_acc = base_TP/base_samples
    genotype_acc = genotype_TP/genotype_samples
    indel1_acc = indel1_TP/indel1_samples
    indel2_acc = indel2_TP/indel2_samples
    acc = (base_acc+genotype_acc+indel1_acc+indel2_acc)/4
    return acc


def lr_finder(lr_accuracy):
    df = pd.DataFrame(lr_accuracy, columns=["lr", "accuracy", "loss"])
    df['diff'] = df['accuracy'].diff()
    df = df.dropna().reset_index(drop=True)
    minimum_lr = df[df['diff'] == max(df['diff'])]['lr'].sort_values(ascending=False).item()
    maximum_lr = df[df['diff'] == min(df['diff'])]['lr'].sort_values(ascending=True).item()
    if minimum_lr > maximum_lr:
        minimum_lr, maximum_lr = maximum_lr, minimum_lr
    return minimum_lr, maximum_lr, df


logging.basicConfig(format='%(message)s', level=logging.INFO)


def is_validation_loss_goes_up_and_down(validation_losses):
    if len(validation_losses) <= 6:
        return False

    return (
        validation_losses[-6][0] > validation_losses[-5][0] and
        validation_losses[-5][0] < validation_losses[-4][0] and
        validation_losses[-4][0] > validation_losses[-3][0] and
        validation_losses[-3][0] < validation_losses[-2][0] and
        validation_losses[-2][0] > validation_losses[-1][0]
    ) or (
        validation_losses[-6][0] < validation_losses[-5][0] and
        validation_losses[-5][0] > validation_losses[-4][0] and
        validation_losses[-4][0] < validation_losses[-3][0] and
        validation_losses[-3][0] > validation_losses[-2][0] and
        validation_losses[-2][0] < validation_losses[-1][0]
    )


def is_last_five_epoch_approaches_minimum(validation_losses):
    if len(validation_losses) <= 5:
        return True

    minimum_validation_loss = min(np.asarray(validation_losses)[:, 0])
    return (
        validation_losses[-5][0] == minimum_validation_loss or
        validation_losses[-4][0] == minimum_validation_loss or
        validation_losses[-3][0] == minimum_validation_loss or
        validation_losses[-2][0] == minimum_validation_loss or
        validation_losses[-1][0] == minimum_validation_loss
    )


def is_validation_losses_keep_increasing(validation_losses):
    if len(validation_losses) <= 6:
        return False

    minimum_validation_loss = min(np.asarray(validation_losses)[:, 0])
    return (
        validation_losses[-5][0] > minimum_validation_loss and
        validation_losses[-4][0] > minimum_validation_loss and
        validation_losses[-3][0] > minimum_validation_loss and
        validation_losses[-2][0] > minimum_validation_loss and
        validation_losses[-1][0] > minimum_validation_loss
    )


def shuffle_first_n_items(array, n):
    if len(array) <= n:
        np.random.shuffle(array)
        return array
    # pylint: disable=unbalanced-tuple-unpacking
    a1, a2 = np.split(array, [n])
    np.random.shuffle(a1)
    return np.append(a1, a2)


def train_model(m, training_config):
    learning_rate = param.min_lr
    l2_regularization_lambda = training_config.l2_regularization_lambda
    output_file_path_prefix = training_config.output_file_path_prefix
    summary_writer = training_config.summary_writer
    model_initalization_file_path = training_config.model_initalization_file_path

    dataset_info = training_config.dataset_info
    dataset_size = dataset_info.dataset_size

    training_losses = []
    validation_losses = []
    lr_accuracy = []

    if model_initalization_file_path is not None:
        m.restore_parameters(os.path.abspath(model_initalization_file_path))

    logging.info("[INFO] Start training...")
    logging.info("[INFO] Learning rate: %.2e" % m.set_learning_rate(learning_rate))
    logging.info("[INFO] L2 regularization lambda: %.2e" % m.set_l2_regularization_lambda(l2_regularization_lambda))

    # Model Constants
    training_start_time = time.time()
    no_of_training_examples = (
        dataset_info.no_of_training_examples_from_train_binary or int(dataset_size * param.trainingDatasetPercentage)
    )
    no_of_validation_examples = dataset_info.dataset_size - no_of_training_examples
    no_of_blosc_blocks = utils.no_of_blosc_blocks_from(
        dataset_info=dataset_info,
        no_of_training_examples=no_of_training_examples,
        blosc_block_size=param.bloscBlockSize
    )
    no_of_training_blosc_blocks = int(no_of_training_examples / param.bloscBlockSize)
    tensor_block_index_list = np.arange(no_of_blosc_blocks, dtype=int)

    total_numbers_of_iterations = np.ceil(no_of_training_examples / param.trainBatchSize+1)
    step_size = param.stepsizeConstant * total_numbers_of_iterations

    # Initialize variables
    epoch_count = 1
    if model_initalization_file_path is not None:
        epoch_count = int(model_initalization_file_path[-param.parameterOutputPlaceHolder:])+1

    epoch_start_time = time.time()
    training_loss_sum = 0
    validation_loss_sum = 0
    data_index = 0
    blosc_index = 0
    first_blosc_block_data_index = 0
    x_batch = None
    y_batch = None
    global_step = 0

    base_change_loss_sum = 0
    genotype_loss_sum = 0
    indel_length_loss_sum_1 = 0
    indel_length_loss_sum_2 = 0
    l2_loss_sum = 0

    while epoch_count <= param.lr_finder_max_epoch:
        is_training = data_index < no_of_training_examples
        is_validation = data_index >= no_of_training_examples
        is_with_batch_data = x_batch is not None and y_batch is not None

        # threads for either train or validation
        thread_pool = []
        if is_with_batch_data and is_training:
            thread_pool.append(Thread(target=m.lr_train, args=(x_batch, y_batch)))
        elif is_with_batch_data and is_validation:
            thread_pool.append(Thread(target=m.get_loss, args=(x_batch, y_batch, True)))
        for t in thread_pool:
            t.start()

        next_x_batch, next_y_batch, next_first_blosc_block_data_index, next_blosc_start_index = utils.new_mini_batch(
            data_index=data_index,
            blosc_start_index=blosc_index,
            first_blosc_block_data_index=first_blosc_block_data_index,
            no_of_training_examples=no_of_training_examples,
            no_of_blosc_blocks=no_of_blosc_blocks,
            dataset_info=dataset_info,
            tensor_block_index_list=tensor_block_index_list,
        )

        # wait until loaded next mini batch & finished training/validation with current mini batch
        for t in thread_pool:
            t.join()

        # add training loss or validation loss
        if is_with_batch_data and is_training:
            training_loss_sum += m.trainLossRTVal
            batch_acc = accuracy(m.predictBaseRTVal, m.predictGenotypeRTVal,
                                 m.predictIndelLengthRTVal1, m.predictIndelLengthRTVal2, y_batch)
            lr_accuracy.append((learning_rate, batch_acc, m.trainLossRTVal))
            if summary_writer is not None:
                summary = m.trainSummaryRTVal
                summary_writer.add_summary(summary, epoch_count)
        elif is_with_batch_data and is_validation:
            validation_loss_sum += m.getLossLossRTVal

            base_change_loss_sum += m.base_change_loss
            genotype_loss_sum += m.genotype_loss
            indel_length_loss_sum_1 += m.indel_length_loss_1
            indel_length_loss_sum_2 += m.indel_length_loss_2
            l2_loss_sum += m.l2_loss

        batch_size = np.shape(next_x_batch)[0]
        data_index += batch_size
        blosc_index = next_blosc_start_index
        first_blosc_block_data_index = next_first_blosc_block_data_index

        # if not go through whole dataset yet, continue the process
        if next_first_blosc_block_data_index >= 0 and next_blosc_start_index >= 0:
            x_batch = next_x_batch
            y_batch = next_y_batch
            learning_rate, global_step, _max_learning_rate = m.clr(
                global_step, step_size, param.max_lr, "tri"
            )
            continue

        logging.info(
            " ".join([str(epoch_count), "Training loss:", str(training_loss_sum/no_of_training_examples)])
        )
        logging.info(
            "\t".join([
                "{} Validation loss (Total/Base/Genotype/Indel_1_2):".format(epoch_count),
                str(validation_loss_sum/no_of_validation_examples),
                str(base_change_loss_sum/no_of_validation_examples),
                str(genotype_loss_sum/no_of_validation_examples),
                str(indel_length_loss_sum_1/no_of_validation_examples),
                str(indel_length_loss_sum_2/no_of_validation_examples)
            ])
        )

        logging.info("[INFO] Epoch time elapsed: %.2f s" % (time.time() - epoch_start_time))
        training_losses.append((training_loss_sum, epoch_count))
        validation_losses.append((validation_loss_sum, epoch_count))

        # Output the model
        if output_file_path_prefix != None:
            parameter_output_path = "%s-%%0%dd" % (output_file_path_prefix, param.parameterOutputPlaceHolder)
            m.save_parameters(os.path.abspath(parameter_output_path % epoch_count))

        # variables update per epoch
        epoch_count += 1
        minimum_lr, maximum_lr, df = lr_finder(lr_accuracy)
        logging.info("[INFO] min_lr: %g, max_lr: %g" % (minimum_lr, maximum_lr))
        df.to_csv("lr_finder.txt", sep=',', index=False)

        epoch_start_time = time.time()
        training_loss_sum = 0
        validation_loss_sum = 0
        data_index = 0
        x_batch = None
        y_batch = None

        base_change_loss_sum = 0
        genotype_loss_sum = 0
        indel_length_loss_sum_1 = 0
        indel_length_loss_sum_2 = 0
        l2_loss_sum = 0

        # shuffle data on each epoch
        tensor_block_index_list = shuffle_first_n_items(tensor_block_index_list, no_of_training_blosc_blocks)
        logging.info("[INFO] Shuffled: " + ' '.join(
            [str(x) for x in np.append(tensor_block_index_list[:5], tensor_block_index_list[-5:])]
        ))

    logging.info("[INFO] Training time elapsed: %.2f s" % (time.time() - training_start_time))
    return training_losses, validation_losses


if __name__ == "__main__":

    random.seed(param.RANDOM_SEED)
    np.random.seed(param.RANDOM_SEED)

    parser = argparse.ArgumentParser(description="Train Clair")

    # binary file path
    parser.add_argument('--bin_fn', type=str, default=None,
                        help="Binary tensor input generated by tensor2Bin.py, tensor_fn, var_fn and bed_fn will be ignored")

    # tensor file path
    parser.add_argument('--tensor_fn', type=str, default="vartensors", help="Tensor input")

    # variant file path
    parser.add_argument('--var_fn', type=str, default="truthvars", help="Truth variants list input")

    # bed file path
    parser.add_argument('--bed_fn', type=str, default=None,
                        help="High confident genome regions input in the BED format")

    # checkpoint file path
    parser.add_argument('--chkpnt_fn', type=str, default=None,
                        help="Input a checkpoint for testing or continue training")

    # learning rate, with default value stated in param
    parser.add_argument('--learning_rate', type=float, default=param.initialLearningRate,
                        help="Set the initial learning rate, default: %(default)s")

    # l2 regularization
    parser.add_argument('--lambd', type=float, default=param.l2RegularizationLambda,
                        help="Set the l2 regularization lambda, default: %(default)s")

    # output checkpint file path prefix
    parser.add_argument('--ochk_prefix', type=str, default=None,
                        help="Prefix for checkpoint outputs at each learning rate change, REQUIRED")

    parser.add_argument('--olog_dir', type=str, default=None,
                        help="Directory for tensorboard log outputs, optional")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    # initialize
    logging.info("[INFO] Initializing")
    utils.setup_environment()
    m = cv.Clair()
    m.init()

    dataset_info = utils.dataset_info_from(
        binary_file_path=args.bin_fn,
        tensor_file_path=args.tensor_fn,
        variant_file_path=args.var_fn,
        bed_file_path=args.bed_fn
    )
    training_config = utils.TrainingConfig(
        dataset_info=dataset_info,
        learning_rate=args.learning_rate,
        l2_regularization_lambda=args.lambd,
        output_file_path_prefix=args.ochk_prefix,
        model_initalization_file_path=args.chkpnt_fn,
        summary_writer=m.get_summary_file_writer(args.olog_dir) if args.olog_dir != None else None,
    )

    _training_losses, validation_losses = train_model(m, training_config)

    # show the parameter set with the smallest validation loss
    validation_losses.sort()
    best_validation_epoch = validation_losses[0][1]
    logging.info("[INFO] Best validation loss at epoch: %d" % best_validation_epoch)

    # load best validation model and evaluate it
    model_file_path = "%s-%%0%dd" % (training_config.output_file_path_prefix, param.parameterOutputPlaceHolder)
    best_validation_model_file_path = model_file_path % best_validation_epoch
    m.restore_parameters(os.path.abspath(best_validation_model_file_path))
    evaluate.evaluate_model(m, dataset_info)
