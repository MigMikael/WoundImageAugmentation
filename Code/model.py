from cntk.device import try_set_default_device, gpu
import numpy as np
import sys
import os
import math
import cntk
import cntk.learners
import time


class ColorCalibrationModel:
    def __init__(self, path_train, path_test):
        try_set_default_device(gpu(0))
        self.path_train = path_train
        self.path_test = path_test
        self.train_file = None
        self.test_file = None
        self.input_dim = None
        self.num_label_classes = None
        self.input = None
        self.label = None
        self.hidden_layers_dim = None
        self.z = None
        self.loss = None
        self.learning_rate = None
        self.lr_schedule = None
        self.learner = None
        self.trainer = None
        self.minibatch_size = None
        self.num_minibatches_to_train = None
        self.reader_train = None
        self.reader_test = None
        self.input_map = None

    def create_reader(self, path, is_training, input_dim, num_label_class):
        label_stream = cntk.io.StreamDef(field='labels', shape=num_label_class, is_sparse=False)
        feature_stream = cntk.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
        deserailizer = cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(labels=label_stream, features=feature_stream))

        return cntk.io.MinibatchSource(deserailizer, randomize=is_training, max_sweeps=cntk.io.INFINITELY_REPEAT if is_training else 1, randomization_window_in_chunks=2646)

    def file_reader(self):
        data_found = False
        for data_dir in ['.']:
            self.train_file = os.path.join(data_dir, self.path_train)
            self.test_file = os.path.join(data_dir, self.path_test)
            if os.path.isfile(self.train_file) and os.path.isfile(self.test_file):
                data_found = True
                break

            if not data_found:
                raise ValueError("Your data file is not available. it should be in Data folder")

        print("Data Dir is {0}".format(data_dir))
        print("Train Data path is " + self.train_file)
        print("Test Data path is " + self.test_file)

    def create_model(self, features, hidden_layers_dim):
        with cntk.layers.default_options(init=cntk.glorot_uniform(), activation=cntk.ops.relu):
            h = cntk.layers.Dense(hidden_layers_dim, activation=cntk.relu)(features)
            h = cntk.layers.BatchNormalization(map_rank=1)(h)
            h = cntk.layers.Dense(hidden_layers_dim, activation=cntk.relu)(h)
            h = cntk.layers.BatchNormalization(map_rank=1)(h)
            h = cntk.layers.Dense(hidden_layers_dim, activation=cntk.relu)(h)
            h = cntk.layers.BatchNormalization(map_rank=1)(h)
            h = cntk.layers.Dense(hidden_layers_dim, activation=cntk.relu)(h)
            h = cntk.layers.BatchNormalization(map_rank=1)(h)
            r = cntk.layers.Dense(self.num_label_classes, activation=None)(h)
            cntk.logging.log_number_of_parameters(r)
            return r

    def config_parameter(self):
        self.input_dim = 3
        self.num_label_classes = 3
        self.input = cntk.input_variable(self.input_dim)
        self.label = cntk.input_variable(self.num_label_classes)

        self.hidden_layers_dim = 9
        self.z = self.create_model(self.input, self.hidden_layers_dim)
        self.loss = cntk.squared_error(self.z, self.label) #

        self.learning_rate = 0.00002
        self.lr_schedule = cntk.learning_rate_schedule(self.learning_rate, cntk.UnitType.minibatch)

        self.learner = cntk.sgd(self.z.parameter, self.lr_schedule)
        self.trainer = cntk.Trainer(self.z, (self.loss, self.loss), [self.learner])

        self.minibatch_size = 200
        num_samples_per_sweep = 259308
        num_sweeps_to_train_with = 10
        self.num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / self.minibatch_size

        self.reader_train = self.create_reader(self.train_file, True, self.input_dim, self.num_label_classes)
        self.input_map = {
            self.label: self.reader_train.streams.labels,
            self.input: self.reader_train.features
        }

    def moving_averate(self, a, w=5):
        if len(a) < w:
            return a[:]
        return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]

    def print_training_progress(self, trainer, mb, frequency, file_name, verbose=1):
        training_loss = "N/A"
        eval_error = "N/A"

        if mb % frequency == 0:
            training_loss = trainer.previous_minibatch_loss_average
            eval_error = trainer.previous_minibatch_evaluation_average
            if verbose:
                print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error * 100))
                with open(file_name, 'a') as outfile:
                    outfile.write(
                        "Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}% \n".format(mb, training_loss, eval_error * 100))

        return mb, training_loss, eval_error

    def run_trainer(self):
        training_progress_output_freq = 500
        np.random.seed(0)

        plotdata = {"batchsize": [], "loss": [], "error": []}

        data_path = './train_result/'
        name = 'result-'
        file_extension = '.txt'
        file_name = data_path + name + str(time.ctime()) + file_extension
        with open(file_name, 'a') as outfile:
            outfile.write("hidden_layer_dim = " + str(self.hidden_layers_dim) + "\n")
            outfile.write("hidden_layer_num = 3\n")
            outfile.write("learning_rate = " + str(self.learning_rate) + "\n")
            outfile.write("minibatch_size = " + str(self.minibatch_size) + "\n")

        for i in range(0, int(self.num_minibatches_to_train)):
            data = self.reader_train.next_minibatch(self.minibatch_size, input_map=self.input_map)
            self.trainer.train_minibatch(data)

            batchsize, loss, error = self.print_training_progress(self.trainer, i, training_progress_output_freq, verbose=1, file_name=file_name)
            if not (loss == "NA" or error == "NA"):
                plotdata["batchsize"].append(batchsize)
                plotdata["loss"].append(loss)
                plotdata["error"].append(error)

    def run_testing(self):
        self.reader_test = self.create_reader(self.test_file, False, self.input_dim, self.num_label_classes)
        test_input_map = {
            self.label: self.reader_test.streams.labels,
            self.input: self.reader_test.streams.features,
        }

        test_minibatch_size = 200
        num_samples = 2646
        num_minibatches_to_test = num_samples // test_minibatch_size
        test_result = 0.0

        for i in range(num_minibatches_to_test):
            data = self.reader_test.next_minibatch(test_minibatch_size, input_map=test_input_map)

            pre_temp = data[self.input].asarray()[0]
            print("Pre --------------------")
            print(pre_temp)

            post_temp = data[self.label].asarray()[0]
            print("Post --------------------")
            print(post_temp)
            print()

            eval_error = self.trainer.test_minibatch(data)
            test_result = test_result + eval_error

        # Average of evaluation errors of all test minibatches
        print("Average test error (ratio error): {0:.2f}%".format((test_result * 100 / num_minibatches_to_test) / num_samples))

    def start(self):
        self.file_reader()
        self.config_parameter()
        self.run_trainer()
        self.run_testing()
