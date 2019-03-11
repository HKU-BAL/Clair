import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import sys
import argparse
import io
import os
import re
import selu
import param
import math
from collections import defaultdict
import multiprocessing

class Clair(object):
    """
    Modification of Clairvoyante v3

    Keywords arguments:

    float_type: The type of float to be used for tensorflow, default tf.float64
    input_shape: Shpae of the input tensor, a tuple or list of 3 integers

    task_loss_weights:
        The weights of different tasks in the calculation of total loss, list of 5 integers in order
        (base_change, zygosity, variant type, indel length, L2 regularization)
    structure: The name of the structure, supporting "FC_L3_narrow_legacy_0.1, 2BiLST, CNN1D_L6, Res1D_L9M"

    output_base_change_shape: The number of classes in the ouput of base change (alternate base) prediction
    output_zygosity_shape: The number of classes in the ouput of zygosity prediction
    output_variant_type_shape:
        The number of classes in the ouput of variant type (No variant, SNP, insertion, deletion, etc.) prediction
    output_indel_length_shape: The number of classes in the ouput of indel length prediction

    output_weight_enabled: True enables per class weights speficied in output_*_entropy_weights (Slower)
    output_base_change_entropy_weights:
        A list of (output_base_change_shape) integers specifying the weights of different classes in
        the calculation of entropy loss (Only used when output_weight_enabled is set to True)
    output_zygosity_entropy_weights, output_variant_type_entropy_weights, output_indel_length_entropy_weights:
        similar to output_base_change_entropy_weights

    L1_num_units: Number of units in L1

    tensor_transform_function:
        the function (callable) for transforming the input tensors to match the model, takes in
        X_tensor, Y_tensor, and stage text ("train" or "predict") and
        return the pair (transformed_X, transformed_Y)
        i.e. type: tensor -> tensor -> str -> (tensor, tensor)
        default: lambda X, Y, phase: (X, Y)  (identity ignoring stage text), which is equivalent to def f(X, Y, phase): return (X, Y)

    """
    COLORS_RGB = dict(
        RED=[1.0, 0.0, 0.0],
        GREEN=[0.0, 1.0, 0.0],
        BLUE=[0.0, 0.0, 1.0],
        WHITE=[1.0, 1.0, 1.0],
        BLACK=[0.0, 0.0, 0.0]
    )

    def __init__(self, **kwargs):

        # Define the default dictionary here
        params = dict(
            float_type=tf.float64,
            input_shape=(2 * param.flankingBaseNum + 1, param.matrixRow, param.matrixNum),
            task_loss_weights=[
                1,                       # base change
                1,                       # zygosity
                1,                       # variant type
                1,                       # indel length
                1                        # l2 loss
            ],
            structure="2BiLSTM",
            output_base_change_shape=4,
            output_zygosity_shape=2,
            output_variant_type_shape=4,
            output_indel_length_shape=6,
            output_base_change_entropy_weights=[
                1,
                1,
                1,
                1
            ],
            output_zygosity_entropy_weights=[
                1,
                1
            ],
            output_variant_type_entropy_weights=[
                1,
                1,
                1,
                1
            ],
            output_indel_length_entropy_weights=[
                1,
                1,
                1,
                1,
                1,
                1
            ],
            L1_num_units=30,
            L2_num_units=30,
            L3_num_units=192,
            L3_dropout_rate=0.5,
            LSTM1_num_units=128,
            LSTM2_num_units=128,
            LSTM3_num_units=128,
            LSTM1_dropout_rate=0,
            LSTM2_dropout_rate=0.5,
            LSTM3_dropout_rate=0.5,
            initial_learning_rate=param.initialLearningRate,
            learning_rate_decay=param.learningRateDecay,
            l2_regularization_lambda=param.l2RegularizationLambda,
            l2_regularization_lambda_decay_rate=param.l2RegularizationLambdaDecay,
            tensor_transform_function=lambda X, Y, phase: (X, Y)
        )

        # Getting other parameters from the param.py file
        params_from_file = param.get_model_parameters()
        params.update(params_from_file)
        for key, value in kwargs.items():
            if key in params.keys():
                params[key] = value
            else:
                print("Info: the parameter %s, with value %s is not supported" % (key, value))
        # print(sorted(params.items()))

        # Extract the values from the params dictionary
        self.input_shape = params['input_shape']
        self.tensor_transform_function = params['tensor_transform_function']
        self.output_base_change_shape = params['output_base_change_shape']
        self.output_zygosity_shape = params['output_zygosity_shape']
        self.output_variant_type_shape = params['output_variant_type_shape']
        self.output_indel_length_shape = params['output_indel_length_shape']

        self.task_loss_weights = np.array(params['task_loss_weights'], dtype=float)

        self.output_base_change_entropy_weights = np.array(params['output_base_change_entropy_weights'], dtype=float)
        self.output_zygosity_entropy_weights = np.array(params['output_zygosity_entropy_weights'], dtype=float)
        self.output_variant_type_entropy_weights = np.array(params['output_variant_type_entropy_weights'], dtype=float)
        self.output_indel_length_entropy_weights = np.array(params['output_indel_length_entropy_weights'], dtype=float)

        self.L1_num_units = params['L1_num_units']
        self.L2_num_units = params['L2_num_units']
        self.L3_num_units = params['L3_num_units']
        self.L3_dropout_rate = params['L3_dropout_rate']

        self.LSTM1_num_units = params['LSTM1_num_units']
        self.LSTM2_num_units = params['LSTM2_num_units']
        self.LSTM3_num_units = params['LSTM3_num_units']
        self.LSTM1_dropout_rate = params['LSTM1_dropout_rate']
        self.LSTM2_dropout_rate = params['LSTM2_dropout_rate']
        self.LSTM3_dropout_rate = params['LSTM3_dropout_rate']

        self.learning_rate_value = params['initial_learning_rate']
        self.learning_rate_decay_rate = params['learning_rate_decay']
        self.l2_regularization_lambda_value = params['l2_regularization_lambda']
        self.l2_regularization_lambda_decay_rate = params['l2_regularization_lambda_decay_rate']
        self.structure = params['structure']

        # Ensure the appropriate float datatype is used for Convolutional / Recurrent networks, which does not support tf.float64
        if 'CNN' in self.structure or 'Res' in self.structure or 'LSTM' in self.structure or 'GRU' in self.structure:
            self.float_type = tf.float32
        else:
            self.float_type = params['float_type']

        self.output_cache = {}

        # Specify the way to split the output ground truth label
        self.output_label_split = [self.output_base_change_shape, self.output_zygosity_shape,
                                   self.output_variant_type_shape, self.output_indel_length_shape]

        # print(self.input_shape)

        tf.set_random_seed(param.RANDOM_SEED)
        self.g = tf.Graph()
        self._build_graph()

        print("[INFO] Using %d CPU threads" % (param.NUM_THREADS))
        self.netcfg = tf.ConfigProto()
        self.netcfg.intra_op_parallelism_threads = param.NUM_THREADS
        self.netcfg.inter_op_parallelism_threads = param.NUM_THREADS

        self.session = tf.Session(
            graph=self.g,
            config=self.netcfg
        )

    @staticmethod
    def get_available_gpus():
        """
        Return the names of gpu units available on the system
        """
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    def get_structure_dict(self, phase='train'):
        """
        A function for getting the appropriate values for placeholders, based on whether the phase is "train" or not

        Return:
            A dictionary containing values for the placeholders
        """
        if phase == 'train':
            return {self.L3_dropout_rate_placeholder: self.L3_dropout_rate}
        else:
            return {self.L3_dropout_rate_placeholder: 0.0}

    @staticmethod
    def slice_dense_layer(inputs, units, slice_dimension, name="slice_dense", **kwargs):
        """
        Specify a slice dense layer, which unpacks along the specified dimension and connects each position to another layer by full connections
        e.g. A tensor of shape [4, 5] would be unpacked to 4 tensors with shape [5], and each of the tensor with shape [5] is fully connected
             to another tensor with [units], and restacked to give a tensor with output shape [4, units]

        inputs: The input tensor
        units: The number of units for each position
        slice_dimension: The index of the dimension to be sliced, following the order of tensor.shape
        name: The name of the operation (variable scope)
        **kwargs: Other parameters to be passed to the tf.layers.dense() function
        """
        with tf.variable_scope(name):
            sliced = tf.unstack(inputs, axis=slice_dimension, name=name + "Unstack")
            slice_dense = tf.stack(
                [tf.layers.dense(v, units=units, name="Unit_" + str(i), **kwargs) for i, v in enumerate(sliced)],
                axis=slice_dimension,
                name="Stacked"
            )
            return slice_dense

    @staticmethod
    def weighted_cross_entropy(softmax_prediction, labels, weights, epsilon, name):
        """
        Compute cross entropy with per class weights

        softmax_prediction: The softmaxed tensor produced by the model, should have shape (batch, number of output classes)
        labels: The output labels in one-hot encoding
        weights: The weights for each class, must have same shape as the number of classes in the output, i.e. the output shape

        Return:
            Tensor representing the weighted cross entropy, having shape of (batch size, )
        """
        return -tf.reduce_sum(
            tf.multiply(
                labels * tf.log(softmax_prediction + epsilon),
                weights
            ),
            reduction_indices=[1],
            name=name
        )

    @staticmethod
    def adaptive_LSTM_layer(inputs, num_units, name="adaptive_LSTM", direction="bidirectional", num_layers=1, cudnn_gpu_available=False):
        """
        A wrapper function for selecting the appropriate LSTM layer to use depending on whether cudnn compatible gpu is available

        Args:
            inputs: Tensor, The input tensor to the LSTM layer, time-major (i.e. in shape (time-steps, batch, sequence))
            num_units: int, The number of units in each direction (i.e. will have a total of 2 * num_units outputs for bidirectional LSTM)
            direction: str, "bidirectional" for bidirectional LSTM, unidirectional otherwise
            num_layers: int, the number of layers stacked together, each having the same number of units
            cudnn_gpu_available: bool, if True, the Cudnn enabled version will be used, otherwise, a compatible version is used

        Return: (outputs, output_states)
            outputs: Tensor, containing the output of the LSTM
            output_states: A tuple of two Tensors for bidirectional LSTM, the first one being the final state for the forward LSTM, and the second one is backward
                           If unidirectional, contains only a single Tensor for the final state of the LSTM
        """
        with tf.variable_scope(name):
            if cudnn_gpu_available:
                lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
                    num_layers=num_layers,
                    num_units=num_units,
                    direction=direction,
                    dtype=tf.float32,
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
                        factor=1.0,
                        mode='FAN_IN',
                        seed=param.OPERATION_SEED
                    ),
                    seed=param.OPERATION_SEED
                )
                lstm.build(inputs.get_shape())
                outputs, output_states = lstm(inputs)
                return outputs, output_states

            # print("[INFO] GPU not available")
            if direction == "bidirectional":
                def single_cell_generator():
                    return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units)
                # , reuse=tf.get_variable_scope().reuse
                lstm_fw_cells = [single_cell_generator() for _ in range(num_layers)]
                lstm_bw_cells = [single_cell_generator() for _ in range(num_layers)]
                (outputs, output_state_fw, output_state_bw) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
                    lstm_fw_cells,
                    lstm_bw_cells,
                    inputs,
                    dtype=tf.float32,
                    time_major=True
                )
                return outputs, (output_state_fw, output_state_bw)
            else:
                def single_cell_generator():
                    return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_units)
                # NOTE: Even if there's only one layer, the cell needs to be wrapped in MultiRNNCell.
                cell = tf.nn.rnn_cell.MultiRNNCell([single_cell_generator() for _ in range(num_layers)])
                # Leave the scope arg unset.
                outputs, final_state = tf.nn.dynamic_rnn(
                    cell,
                    inputs,
                    dtype=tf.float32,
                    time_major=True
                )
                return outputs, final_state

    def _build_graph(self):
        """
        Build the computation graph for the model
        """

        self.graph = self.g
        self.layers = []  # A list used to contain meaningful intermediate layers
        with self.graph.as_default():
            tf.set_random_seed(param.RANDOM_SEED)

            # Conversion to tensors for some values
            self.epsilon = tf.constant(value=1e-10, dtype=self.float_type)
            self.input_shape_tf = (None, self.input_shape[0], self.input_shape[1], self.input_shape[2])

            # Place holders
            self.X_placeholder = tf.placeholder(self.float_type, self.input_shape_tf, name='X_placeholder')
            self.Y_placeholder = tf.placeholder(
                self.float_type,
                [None, self.output_base_change_shape + self.output_zygosity_shape +
                    self.output_variant_type_shape + self.output_indel_length_shape],
                name='Y_placeholder'
            )
            self.layers.append(self.X_placeholder)

            self.learning_rate_placeholder = tf.placeholder(self.float_type, shape=[], name='learning_rate_placeholder')
            self.phase_placeholder = tf.placeholder(tf.bool, shape=[], name='phase_placeholder')
            self.regularization_L2_lambda_placeholder = tf.placeholder(
                self.float_type, shape=[], name='regularization_L2_lambda_placeholder')
            self.task_loss_weights_placeholder = tf.placeholder(
                self.float_type, shape=self.task_loss_weights.shape, name='task_loss_weights_placeholder')
            self.output_base_change_entropy_weights_placeholder = tf.placeholder(
                self.float_type, shape=self.output_base_change_entropy_weights.shape, name='output_base_change_entropy_weights_placeholder')
            self.output_zygosity_entropy_weights_placeholder = tf.placeholder(
                self.float_type, shape=self.output_zygosity_entropy_weights.shape, name='output_zygosity_entropy_weights_placeholder')
            self.output_variant_type_entropy_weights_placeholder = tf.placeholder(
                self.float_type, shape=self.output_variant_type_entropy_weights.shape, name='output_variant_type_entropy_weights_placeholder')
            self.output_indel_length_entropy_weights_placeholder = tf.placeholder(
                self.float_type, shape=self.output_indel_length_entropy_weights.shape, name='output_indel_length_entropy_weights_placeholder')

            he_initializer = tf.contrib.layers.variance_scaling_initializer(
                factor=1.0,
                mode='FAN_IN',
                seed=param.OPERATION_SEED
            )

            if self.structure == "2BiLSTM":
                self.L3_dropout_rate_placeholder = tf.placeholder(
                    self.float_type, shape=[], name='L3_dropout_rate_placeholder')

                # Flatten the 2nd (ACGT) and 3rd (Ref Ins Del SNP) dimension
                self.X_flattened_2D = tf.reshape(
                    self.X_placeholder,
                    shape=(
                        tf.shape(self.X_placeholder)[0],
                        self.input_shape_tf[1],
                        self.input_shape_tf[2] * self.input_shape_tf[3]
                    ),
                    name="X_flattened_2D"
                )
                self.layers.append(self.X_flattened_2D)

                self.X_flattened_2D_transposed = tf.transpose(
                    self.X_flattened_2D, [1, 0, 2], name="X_flattened_2D_transposed")

                is_gpu_available = len(Clair.get_available_gpus()) > 0
                # print("is_gpu_available:", is_gpu_available)
                self.LSTM1, self.LSTM1_state = Clair.adaptive_LSTM_layer(
                    inputs=self.X_flattened_2D_transposed,
                    num_units=self.LSTM1_num_units,
                    name="LSTM1",
                    direction="bidirectional",
                    num_layers=1,
                    cudnn_gpu_available=is_gpu_available
                )
                self.layers.append(self.LSTM1)
                # print(self.LSTM1, self.LSTM1_state)
                self.LSTM1_dropout = tf.layers.dropout(
                    inputs=self.LSTM1,
                    rate=self.LSTM1_dropout_rate,
                    training=self.phase_placeholder,
                    name="LSTM1_dropout",
                    seed=param.OPERATION_SEED
                )
                self.LSTM2, _ = Clair.adaptive_LSTM_layer(
                    inputs=self.LSTM1_dropout,
                    num_units=self.LSTM2_num_units,
                    name="LSTM2",
                    direction="bidirectional",
                    num_layers=1,
                    cudnn_gpu_available=is_gpu_available
                )
                self.layers.append(self.LSTM2)
                self.LSTM2_dropout = tf.layers.dropout(
                    inputs=self.LSTM2,
                    rate=self.LSTM2_dropout_rate,
                    training=self.phase_placeholder,
                    name="LSTM2_dropout",
                    seed=param.OPERATION_SEED
                )
                self.LSTM2_transposed = tf.transpose(self.LSTM2_dropout, [1, 0, 2], name="LSTM2_transposed")

                # Slice dense layer 2
                self.L2 = Clair.slice_dense_layer(
                    inputs=self.LSTM2_transposed,
                    units=self.L2_num_units,
                    slice_dimension=2,
                    name="L2",
                    activation=selu.selu,
                    kernel_initializer=he_initializer
                )
                self.layers.append(self.L2)

                self.L2_flattened = tf.reshape(
                    self.L2,
                    shape=(tf.shape(self.L2)[0], self.L2_num_units * self.LSTM2_num_units * 2),
                    name="L2_flattened"
                )
                self.layers.append(self.L2_flattened)

                # Dense layer 3
                self.L3 = tf.layers.dense(
                    inputs=self.L2_flattened,
                    units=self.L3_num_units,
                    name="L3",
                    activation=selu.selu,
                    kernel_initializer=he_initializer
                )
                self.layers.append(self.L3)

                self.L3_dropout = selu.dropout_selu(
                    self.L3,
                    self.L3_dropout_rate_placeholder,
                    training=self.phase_placeholder,
                    name='L3_dropout',
                    seed=param.OPERATION_SEED
                )
                self.layers.append(self.L3_dropout)

                self.core_final_layer = self.L3_dropout

            # Output layer
            with tf.variable_scope("Prediction"):
                self.Y_base_change_logits = tf.layers.dense(
                    inputs=self.core_final_layer,
                    units=self.output_base_change_shape,
                    kernel_initializer=he_initializer,
                    activation=selu.selu,
                    name='Y_base_change_logits'
                )
                self.Y_base_change = tf.nn.softmax(self.Y_base_change_logits, name='Y_base_change')
                self.layers.append(self.Y_base_change)

                self.Y_zygosity_logits = tf.layers.dense(
                    inputs=self.core_final_layer,
                    units=self.output_zygosity_shape,
                    kernel_initializer=he_initializer,
                    activation=selu.selu,
                    name='Y_zygosity_logits'
                )
                self.Y_zygosity = tf.nn.softmax(self.Y_zygosity_logits, name='Y_zygosity')
                self.layers.append(self.Y_zygosity)

                if "legacy_0.1" in self.structure:
                    self.Y_variant_type_logits = tf.layers.dense(
                        inputs=self.core_final_layer,
                        units=self.output_variant_type_shape,
                        kernel_initializer=he_initializer,
                        activation=selu.selu,
                        name='Y_variant_logits'
                    )
                    self.Y_variant_type = tf.nn.softmax(self.Y_variant_type_logits, name='Y_variant')
                else:
                    self.Y_variant_type_logits = tf.layers.dense(
                        inputs=self.core_final_layer,
                        units=self.output_variant_type_shape,
                        kernel_initializer=he_initializer,
                        activation=selu.selu,
                        name='Y_variant_type_logits'
                    )
                    self.Y_variant_type = tf.nn.softmax(self.Y_variant_type_logits, name='Y_variant_type')
                self.layers.append(self.Y_variant_type)

                self.Y_indel_length_logits = tf.layers.dense(
                    inputs=self.core_final_layer,
                    units=self.output_indel_length_shape,
                    kernel_initializer=he_initializer,
                    activation=selu.selu,
                    name='Y_indel_length_logits'
                )
                self.Y_indel_length = tf.nn.softmax(self.Y_indel_length_logits, name='Y_indel_length')
                self.layers.append(self.Y_indel_length)

                self.Y = [self.Y_base_change, self.Y_zygosity, self.Y_variant_type, self.Y_indel_length]

            # Extract the truth labels by output ratios
            with tf.variable_scope("Loss"):
                Y_base_change_label, Y_zygosity_label, Y_variant_type_label, Y_indel_length_label = tf.split(
                    self.Y_placeholder, self.output_label_split, axis=1, name="label_split")

                # Cross Entropy loss
                Y_variant_type_str = "Y_variant_type"

                self.Y_base_change_cross_entropy = Clair.weighted_cross_entropy(
                    softmax_prediction=self.Y_base_change,
                    labels=Y_base_change_label,
                    weights=self.output_base_change_entropy_weights_placeholder,
                    epsilon=self.epsilon,
                    name="Y_base_change_cross_entropy"
                )
                self.Y_zygosity_cross_entropy = Clair.weighted_cross_entropy(
                    softmax_prediction=self.Y_zygosity,
                    labels=Y_zygosity_label,
                    weights=self.output_zygosity_entropy_weights_placeholder,
                    epsilon=self.epsilon,
                    name="Y_zygosity_cross_entropy"
                )
                self.Y_variant_type_cross_entropy = Clair.weighted_cross_entropy(
                    softmax_prediction=self.Y_variant_type,
                    labels=Y_variant_type_label,
                    weights=self.output_variant_type_entropy_weights_placeholder,
                    epsilon=self.epsilon,
                    name=Y_variant_type_str + "_cross_entropy"
                )
                self.Y_indel_length_entropy = Clair.weighted_cross_entropy(
                    softmax_prediction=self.Y_indel_length,
                    labels=Y_indel_length_label,
                    weights=self.output_indel_length_entropy_weights_placeholder,
                    epsilon=self.epsilon,
                    name="Y_indel_length_entropy"
                )

                self.Y_base_change_loss = tf.reduce_sum(self.Y_base_change_cross_entropy, name="Y_base_change_loss")
                self.Y_zygosity_loss = tf.reduce_sum(self.Y_zygosity_cross_entropy, name="Y_zygosity_loss")
                self.Y_variant_type_loss = tf.reduce_sum(
                    self.Y_variant_type_cross_entropy, name=Y_variant_type_str + "_loss")
                self.Y_indel_length_loss = tf.reduce_sum(self.Y_indel_length_entropy, name="Y_indel_length_loss")

                self.regularization_L2_loss_without_lambda = tf.add_n([
                    tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name
                ])
                self.regularization_L2_loss = (
                    self.regularization_L2_loss_without_lambda * self.regularization_L2_lambda_placeholder
                )

                # Weighted average of losses, speicified by loss_ratio
                self.total_loss = tf.reduce_sum(
                    tf.multiply(
                        self.task_loss_weights_placeholder,
                        tf.stack([
                            self.Y_base_change_loss,
                            self.Y_zygosity_loss,
                            self.Y_variant_type_loss,
                            self.Y_indel_length_loss,
                            self.regularization_L2_loss
                        ])
                    ),
                    name="Total_loss"
                )

            # Create the saver for the model
            self.saver = tf.train.Saver(max_to_keep=1000000,)

            # Include gradient clipping if RNN architectures are used
            if "RNN" in self.structure or "LSTM" in self.structure:
                with tf.variable_scope("Training_Operation"):
                    self.optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate_placeholder
                    )
                    gradients, variables = zip(*self.optimizer.compute_gradients(self.total_loss))
                    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                    self.training_op = self.optimizer.apply_gradients(zip(gradients, variables))
            else:
                self.training_op = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate_placeholder
                ).minimize(self.total_loss)

            self.init_op = tf.global_variables_initializer()

            # Summary logging
            self.training_summary_op = tf.summary.merge([
                tf.summary.scalar('learning_rate', self.learning_rate_placeholder),
                tf.summary.scalar('l2_Lambda', self.regularization_L2_lambda_placeholder),
                tf.summary.scalar("Y_base_change_loss", self.Y_base_change_loss),
                tf.summary.scalar("Y_zygosity_loss", self.Y_zygosity_loss),
                tf.summary.scalar("Y_variant_type_loss", self.Y_variant_type_loss),
                tf.summary.scalar("Y_indel_length_loss", self.Y_indel_length_loss),
                tf.summary.scalar("Regularization_loss", self.regularization_L2_loss),
                tf.summary.scalar("Total_loss", self.total_loss)
            ])

            # For report or debug. Fetching histogram summary is slow, GPU utilization will be low if enabled.
            # for var in tf.trainable_variables():
            #    tf.summary.histogram(var.op.name, var)
            # self.merged_summary_op = tf.summary.merge_all()

            # Aliasing
            self.loss = self.total_loss

            # Getting the total number of traininable parameters
            #total_parameters = 0
            #for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                #shape = variable.get_shape()
                # print(variable.name, shape)
                # print(len(shape))
                #variable_parameters = 1
                #try:
                    #for dim in shape:
                        # print(dim)
                        #variable_parameters *= dim.value
                    #total_parameters += variable_parameters
                #except ValueError as ve:
                    # if the shape cannot be obtained, (e.g. opaque operators)
                    #print("Variable {:s} has unknown shape.".format(variable.name))
                    #print(ve.message)
                # print(variable_parameters)

            #print("Total Trainable Parameters: " + str(total_parameters))

    def init(self):
        """
        Initialize the model by running the init_op and create the summary writer
        """
        # self.current_summary_writer = tf.summary.FileWriter('logs', self.session.graph)
        # print("Preparing to run init")
        self.session.run(self.init_op)

    def get_summary_op_factory(self, render_function, name="Render", *args_factory, **kwargs_factory):
        """
        (Experimental, unstable when using with matplotlib)
        Wrap the rendering function as a tensorflow operation
        """

        def _get_tensor_render_op(in_tensor, *args_func, **kwargs_func):

            def _render_function_wrap(in_tensor, *args):
                img_arrays = [render_function(matrix, *args, **kwargs_func) for matrix in in_tensor]
                return np.stack(img_arrays, axis=0)
                # img_array = render_function(*args, **kwargs_func)
                # return np.expand_dims(img_array, axis=0)
            return tf.py_func(_render_function_wrap, [in_tensor] + list(args_func), Tout=tf.uint8, name="Plot")

        def _summary_op_factory(summary_name, in_tensor, *args_call, **kwargs_call):
            tf_render_op = _get_tensor_render_op(in_tensor, *args_call, **kwargs_call)

            # with tf.name_scope("Batch_Plot"):
            #     unstack_layer = tf.unstack(in_tensor, name='unstack')
            #     tensor_render_ops = [_get_tensor_render_op(slice_tensor, *args_call, **kwargs_call) for slice_tensor in unstack_layer]
            #     image_stacked = tf.stack(tensor_render_ops, name='stack_images')

            return tf.summary.image(summary_name, tf_render_op,
                                    max_outputs=kwargs_call.pop('max_outputs', 3),
                                    collections=kwargs_call.pop('collections', None),
                                    )
        return _summary_op_factory

    @staticmethod
    def recursive_process_tensor(tensor, apply_function, recursion_text="", target_ndim=2, last_first=False, sparator="-", *args, **kwargs):
        """
        A general function processing tensors if they have larger dimension than the target ndim, calling the apply_function for each sliced tensor and
        group the output in a list

        Arguments:
            tensor: Numpy Array, the tensor to be processed
            apply_function: a function to be called for a tensor with the target_ndim
            recursion_text: str, used internally, where in each round, the position of the corresponding matrix is appending to this string, together with a separator
                            e.g. a seed of "ABC" will become "ABC-2" in the next layer for position 2 and separator -
            target_ndim: int, the target number of dimensions to stop the recursion and call the function
            last_first: bool, expand the last dimension first
            sparator: str, for appending when each dimension is processed
            *args, **kwargs: other arguments to be passed to the function "apply_function"

        Returns:
            A list containing all the results from apply_function(sliced_tensor)
        """

        if tensor.ndim <= target_ndim:
            return [apply_function(tensor, recursion_text, *args, **kwargs)]
        else:
            if last_first:
                rolled_tensor = np.rollaxis(tensor, -1)
            recursion_text += sparator
            processed = [Clair.recursive_process_tensor(subtensor, apply_function, recursion_text + str(i), target_ndim=target_ndim, last_first=last_first, *args, **kwargs)
                         for i, subtensor in enumerate(rolled_tensor)]
            return [item for sublist in processed for item in sublist]

    def close(self):
        """
        Closes the current tf session
        """
        self.session.close()

    def train(self, batchX, batchY, result_caching=False):
        """
        Train the model in batch with input tensor batchX and truth tensor batchY, caching the results in
        self.output_cache['training_loss'] and self.output_cache['training_summary'] if result_caching is True
        The tensor transform function is applied prior to training

        Returns:
            loss: The loss value from the batch
            summary: The tf.summary of the training
        """
        # for i in range(len(batchX)):
        #    tf.image.per_image_standardization(batchX[i])
        transformed_batch_X, transformed_batch_Y = self.tensor_transform_function(batchX, batchY, "train")
        input_dictionary = {
            self.X_placeholder: transformed_batch_X,
            self.Y_placeholder: transformed_batch_Y,
            self.learning_rate_placeholder: self.learning_rate_value,
            self.phase_placeholder: True,
            self.regularization_L2_lambda_placeholder: self.l2_regularization_lambda_value,
            self.task_loss_weights_placeholder: self.task_loss_weights,
            self.output_base_change_entropy_weights_placeholder: self.output_base_change_entropy_weights,
            self.output_zygosity_entropy_weights_placeholder: self.output_zygosity_entropy_weights,
            self.output_variant_type_entropy_weights_placeholder: self.output_variant_type_entropy_weights,
            self.output_indel_length_entropy_weights_placeholder: self.output_indel_length_entropy_weights
        }
        input_dictionary.update(self.get_structure_dict(phase='train'))

        loss, _, summary = self.session.run(
            (self.loss, self.training_op, self.training_summary_op),
            feed_dict=input_dictionary
        )
        # if result_caching:
        self.output_cache['training_loss'] = loss
        self.output_cache['training_summary'] = summary

        # Aliasing
        # self.trainLossRTVal = self.output_cache['training_loss']
        # self.trainSummaryRTVal = self.output_cache['training_summary']
        self.trainLossRTVal = loss
        self.trainSummaryRTVal= summary

        return loss, summary

    def predict(self, batchX, result_caching=False):
        """
        Predict using model in batch with input tensor batchX, caching the results in
        self.output_cache['prediction_base'], self.output_cache['prediction_zygosity'], self.output_cache['prediction_variant_type'], self.output_cache['prediction_indel_length']
        if result_caching is True
        The tensor transform function is applied prior to prediction

        Returns:
            base, zygosity, variant_type, indel_length: The four softmaxed prediction from the model in batch
        """
        # for i in range(len(batchX)):
        #    tf.image.per_image_standardization(XArray[i])
        transformed_batch_X, _ = self.tensor_transform_function(batchX, None, "predict")

        input_dictionary = {
            self.X_placeholder: transformed_batch_X,
            self.learning_rate_placeholder: 0.0,
            self.phase_placeholder: False,
            self.regularization_L2_lambda_placeholder: 0.0
        }
        input_dictionary.update(self.get_structure_dict(phase='predict'))

        base, zygosity, variant_type, indel_length = self.session.run(self.Y, feed_dict=input_dictionary)

        if result_caching:
            self.output_cache['prediction_base'] = base
            self.output_cache['prediction_zygosity'] = zygosity
            self.output_cache['prediction_variant_type'] = variant_type
            self.output_cache['prediction_indel_length'] = indel_length

            # Aliasing
            self.predictBaseRTVal = self.output_cache['prediction_base']
            self.predictZygosityRTVal = self.output_cache['prediction_zygosity']
            self.predictVarTypeRTVal = self.output_cache['prediction_variant_type']
            self.predictIndelLengthRTVal = self.output_cache['prediction_indel_length']

        return base, zygosity, variant_type, indel_length

    def get_loss(self, batchX, batchY, result_caching=False):
        """
        Getting the loss using model in batch with input tensor batchX and truth tensor batchY, caching the results in
        self.output_cache['prediction_loss'] if result_caching is True
        The tensor transform function is applied prior to getting loss

        Returns:
            loss: The loss value for this batch
        """
        transformed_batch_X, transformed_batch_Y = self.tensor_transform_function(batchX, batchY, "predict")
        input_dictionary = {
            self.X_placeholder: transformed_batch_X,
            self.Y_placeholder: transformed_batch_Y,
            self.learning_rate_placeholder: 0.0,
            self.phase_placeholder: False,
            self.regularization_L2_lambda_placeholder: 0.0,
            self.task_loss_weights_placeholder: self.task_loss_weights,
            self.output_base_change_entropy_weights_placeholder: self.output_base_change_entropy_weights,
            self.output_zygosity_entropy_weights_placeholder: self.output_zygosity_entropy_weights,
            self.output_variant_type_entropy_weights_placeholder: self.output_variant_type_entropy_weights,
            self.output_indel_length_entropy_weights_placeholder: self.output_indel_length_entropy_weights
        }
        input_dictionary.update(self.get_structure_dict(phase='predict'))

        loss, base_change_loss, zygosity_loss, variant_type_loss, indel_length_loss, l2_loss = self.session.run([
            self.loss,
            self.Y_base_change_loss,
            self.Y_zygosity_loss,
            self.Y_variant_type_loss,
            self.Y_indel_length_loss,
            self.regularization_L2_loss_without_lambda
        ], feed_dict=input_dictionary)

        if result_caching:
            self.output_cache['prediction_loss'] = loss

            # Aliasing
            self.getLossLossRTVal = self.output_cache['prediction_loss']

            self.base_change_loss = base_change_loss
            self.zygosity_loss = zygosity_loss
            self.variant_type_loss = variant_type_loss
            self.indel_length_loss = indel_length_loss
            self.l2_loss = l2_loss * param.l2RegularizationLambda
        return loss

    def save_parameters(self, file_name):
        """
        Save the parameters (weights) to the specific file (file_name)
        """
        self.saver.save(self.session, file_name)

    def restore_parameters(self, file_name):
        """
        Restore the parameters (weights) from the specific file (file_name)
        """
        self.saver.restore(self.session, file_name)

    def get_variable_objects(self, regular_expression):
        """
        Get all variable objects from the graph matching the regular expression

        Returns:
            variable_list: list of tf variable objects
        """
        regex = re.compile(regular_expression)
        variable_list = []
        with self.graph.as_default():
            tf.set_random_seed(param.RANDOM_SEED)
            for variable in tf.trainable_variables():
                if regex.match(variable.name):
                    variable_list.append(variable)
        return variable_list

    def get_operation_objects(self, regular_expression, exclude_expression=".*(grad|tags|Adam).*"):
        """
        Get all operation objects from the graph matching the regular expression, but not the exclude_expression

        Returns:
            operation_list: list of tf operation objects
        """
        regex = re.compile(regular_expression)
        regex_exclude = re.compile(exclude_expression)
        operation_list = []

        for op in self.graph.get_operations():
            if regex.match(op.name) and not regex_exclude.match(op.name):
                print(op.name)
                operation_list.append(op)
        return operation_list

    def get_summary_file_writer(self, logs_path):
        """
        Generate a new tf summary File writer with the specified log path

        returns: A tf.summary.FileWriter object
        """
        # if hasattr(self, "current_summary_writer"):
        #     self.current_summary_writer.close()
        # self.current_summary_writer = tf.summary.FileWriter(logs_path, graph=self.graph)
        # return self.current_summary_writer
        return None

    def set_task_loss_weights(self, task_loss_weights=[1, 1, 1, 1, 1]):
        """
        Assign a set new task loss weights for training

        Arguments:
            task_loss_weights: A list of 5 numbers specifying the weights to the tasks
        """
        self.task_loss_weights = np.array(task_loss_weights, dtype=float)

    def set_learning_rate(self, learning_rate):
        """
        Assign a new learning rate
        """
        self.learning_rate_value = learning_rate
        return self.learning_rate_value

    def decay_learning_rate(self):
        """
        Decay the learning rate by the predefined decay rate
        """
        self.learning_rate_value = self.learning_rate_value * self.learning_rate_decay_rate
        return self.learning_rate_value

    def set_l2_regularization_lambda(self, l2_regularization_lambda):
        """
        Assign a new l2_regularization_lambda value
        """
        self.l2_regularization_lambda_value = l2_regularization_lambda
        return self.l2_regularization_lambda_value

    def decay_l2_regularization_lambda(self):
        """
        Decay the l2_regularization_lambda value by the predefined decay rate
        """
        self.l2_regularization_lambda_value = self.l2_regularization_lambda_value * self.l2_regularization_lambda_decay_rate
        return self.l2_regularization_lambda_value

    def pretty_print_variables(self, regular_expression):
        variable_list = self.get_variable_objects(regular_expression)
        result_string_list = []
        for v in variable_list:
            variable_value = self.session.run(v)
            result_string_list.append(v.name)
            result_string_list.append(Clair.pretty_print_np_tensor(variable_value) + '\n')
        return '\n'.join(result_string_list)

    @staticmethod
    def pretty_print_np_tensor(tensor, element_separator='\t'):
        """
        Print a numpy array (tensor) formatted with [], new lines and the element_separator

        Returns:
            A string containing the formatted tensor
        """
        if tensor.ndim == 1:
            return element_separator.join(('%.16f') % value for value in tensor)
        elif tensor.ndim == 2:
            return_list = []
            for row in tensor:
                return_list.append(Clair.pretty_print_np_tensor(row, element_separator=element_separator))
            return '\n'.join(return_list)
        else:
            return_list = []
            for sub_tensor in tensor:
                return_list.append('[\n' + Clair.pretty_print_np_tensor(sub_tensor,
                                                                            element_separator=element_separator) + '\n]')
            return '\n'.join(return_list)

    def __del__(self):
        # if hasattr(self, "current_summary_writer"):
        #     self.current_summary_writer.close()
        self.session.close()


class FunctionCallConsumer(multiprocessing.Process):
    """
    A class implementing thread safe consumer which does a function call for each task

    Init Arguments:
        target_function: callable, when a task is obtained from the task_queue, the fucntion is called in the args and kwargs from the queue
        task_queue: the task queue, recommend using multiprocessing.JoinableQueue(), each object put into this queue should be a tuple of size 3:
                    (identity, args, kwargs). The identity is only used for identifying the result of the task, and won't be passed to the function
        result_dict: The result dictionary, where the result is put as result_dict[identity] = f(*args, **kwargs)
        name: name of the consumer, for printing
        verbose: printing out message if True
    """

    def __init__(self, target_function, task_queue, result_dict, name="c", verbose=False):
        multiprocessing.Process.__init__(self)
        self.target_function = target_function
        self.task_queue = task_queue
        self.result_dict = result_dict
        self.name = name
        self.verbose = verbose

    def run(self):
        """
        Start the consumer, the consumer stops whenever a None value is put into the queue
        """
        if self.verbose:
            print("Consumer {:s} is starting.".format(self.name))
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                self.task_queue.task_done()
                break

            # identity, f, args, kwargs = next_task["identity"], next_task["f"], next_task["args"], next_task["kwargs"]
            # identity, f, args, kwargs = next_task
            identity, args, kwargs = next_task
            # answer = f(*args, **kwargs)
            answer = self.target_function(*args, **kwargs)
            self.task_queue.task_done()
            # self.result_queue.put((identity, answer))
            self.result_dict[identity] = (identity, answer)
            if self.verbose:
                print("Consumer {:s} finished".format(self.name), identity)
        if self.verbose:
            print("Consumer {:s} is terminating.".format(self.name))
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clair Model")

    parser.add_argument('-v', '--variables', type=str, default=None,
                        help="Print variables matching the regular expression. default: %(default)s")

    parser.add_argument('-r', '--restore_model', type=str, default=None,
                        help="The path to the model to be restored. default: %(default)s")

    parser.add_argument('-s', '--save_model', type=str, default=None,
                        help="The path where the model is to be saved. default: %(default)s")

    parser.add_argument('-l', '--log_dir', type=str, default="logs",
                        help="The path to the log directory. default: %(default)s")

    args = parser.parse_args()

    if args.variables is not None:
        q = Clair()
        q.init()
        if args.restore_model is not None:
            q.restore_parameters(os.path.abspath(args.restore_model))
        print(q.pretty_print_variables(args.variables))
        sys.exit(0)
