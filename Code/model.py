from cntk.device import try_set_default_device, gpu
import numpy as np
import os
import cntk
import cntk.learners
import math
import matplotlib.pyplot as plt


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
        self.num_train_samples_per_sweep = None
        self.num_minibatches_to_train = None
        self.reader_train = None
        self.reader_test = None
        self.input_map = None
        self.file_name = None
        self.num_test_samples = None

    def create_reader(self, path, is_training, input_dim, num_label_class, random_in_chunks):
        label_stream = cntk.io.StreamDef(field='labels', shape=num_label_class, is_sparse=False)
        feature_stream = cntk.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
        deserailizer = cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(labels=label_stream, features=feature_stream))

        return cntk.io.MinibatchSource(deserailizer, randomize=is_training, max_sweeps=cntk.io.INFINITELY_REPEAT if is_training else 1, randomization_window_in_chunks=random_in_chunks)

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

        #print("Data Dir is {0}".format(data_dir))
        #print("Train Data path is " + self.train_file)
        #print("Test Data path is " + self.test_file)

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
            #cntk.logging.log_number_of_parameters(r)
            return r

    def config_parameter(self, hidden_layers_dim, learning_rate, minibatch_size, num_train_samples_per_sweep, num_test_samples, result_file_name):
        self.input_dim = 3
        self.num_label_classes = 3
        self.input = cntk.input_variable(self.input_dim)
        self.label = cntk.input_variable(self.num_label_classes)

        self.hidden_layers_dim = hidden_layers_dim
        self.z = self.create_model(self.input, self.hidden_layers_dim)
        self.loss = cntk.squared_error(self.z, self.label) #

        self.learning_rate = [0.0002]*5000000 + [0.00002]*4000000 + [0.00001]*2000000 + [0.000005]*916288
        #self.learning_rate = learning_rate
        self.lr_schedule = cntk.learning_rate_schedule(self.learning_rate, cntk.UnitType.minibatch)

        self.learner = cntk.sgd(self.z.parameters, self.lr_schedule)
        self.trainer = cntk.Trainer(self.z, (self.loss, self.loss), [self.learner])

        self.minibatch_size = minibatch_size
        self.num_train_samples_per_sweep = num_train_samples_per_sweep
        num_sweeps_to_train_with = 10
        self.num_minibatches_to_train = (self.num_train_samples_per_sweep * num_sweeps_to_train_with) / self.minibatch_size
        
        self.num_test_samples = num_test_samples
        self.file_name = result_file_name     # Name of result file

        self.reader_train = self.create_reader(self.train_file, True, self.input_dim, self.num_label_classes, self.num_train_samples_per_sweep)
        self.input_map = {
            self.label: self.reader_train.streams.labels,
            self.input: self.reader_train.streams.features
        }
    
    '''
    def moving_average(self, a, w=5):
        if len(a) < w:
            return a[:]
        return [val if idx < w else sum(a[(idx - w):idx]) / w for idx, val in enumerate(a)]
    '''

    def print_training_progress(self, trainer, mb, frequency, file_name, verbose=1):
        training_loss = "N/A"
        eval_error = "N/A"

        if mb % frequency == 0:
            training_loss = trainer.previous_minibatch_loss_average
            eval_error = trainer.previous_minibatch_evaluation_average
            if verbose:
                #print("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error * 100))
                with open(file_name, 'a') as outfile:
                    outfile.write("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}% \n".format(mb, training_loss, eval_error * 100))

        return mb, training_loss, eval_error

    def run_trainer(self):
        training_progress_output_freq = 500
        np.random.seed(0)

        plotdata = {"batchsize": [], "loss": [], "error": []}

        with open(self.file_name, 'a') as outfile:
            outfile.write("hidden_layer_dim = " + str(self.hidden_layers_dim) + "\n")
            #outfile.write("learning_rate = " + str(self.learning_rate) + "\n")
            outfile.write("learning_rate = " + "[0.0002]*5000000 + [0.00002]*4000000 + [0.00001]*2000000 + [0.000005]*916288" + "\n")
            outfile.write("minibatch_size = " + str(self.minibatch_size) + "\n")
            outfile.write("num_train_samples_per_sweep = " + str(self.num_train_samples_per_sweep) + "\n")
            outfile.write("num_test_samples = " + str(self.num_test_samples) + "\n")

        for i in range(0, int(self.num_minibatches_to_train)):
            data = self.reader_train.next_minibatch(self.minibatch_size, input_map=self.input_map)
            self.trainer.train_minibatch(data)

            batchsize, loss, error = self.print_training_progress(self.trainer, i, training_progress_output_freq, verbose=1, file_name=self.file_name)
            if not (loss == "NA" or error == "NA"):
                if i % 5000 == 0:
                    plotdata["batchsize"].append(batchsize)
                    plotdata["loss"].append(loss)
                    plotdata["error"].append(error)
                
        #plotdata["avgloss"] = self.moving_average(plotdata["loss"])
        #plotdata["avgerror"] = self.moving_average(plotdata["error"])
        
        plt.figure(num=1, figsize=(12,9), dpi=200)
        
        plt.subplot(211)
        plt.plot(plotdata["batchsize"], plotdata["loss"], 'b--')
        plt.xlabel('Minibatch number')
        plt.ylabel('Loss')
        plt.title('Minibatch run vs. Training loss')
        #plt.show()

        plt.subplot(212)
        plt.plot(plotdata["batchsize"], plotdata["error"], 'r--')
        plt.xlabel('Minibatch number')
        plt.ylabel('Label Prediction Error')
        plt.title('Minibatch run vs. Label Prediction Error')
        #plt.show()
        fig_name = self.file_name.replace("txt", "")
        plt.savefig(fig_name + '_graph.png')

    def run_tester(self):
        self.reader_test = self.create_reader(self.test_file, False, self.input_dim, self.num_label_classes, self.num_test_samples)
        test_input_map = {
            self.label: self.reader_test.streams.labels,
            self.input: self.reader_test.streams.features,
        }

        test_minibatch_size = 1
        num_minibatches_to_test = self.num_test_samples // test_minibatch_size

        max_R = 0
        max_G = 0
        max_B = 0
        
        sum_diff_R = 0
        sum_diff_G = 0
        sum_diff_B = 0
        
        diff_R_list = []
        diff_G_list = []
        diff_B_list = []
        
        
        max_LF_R = 0
        max_LF_G = 0
        max_LF_B = 0
        
        sum_LF_diff_R = 0
        sum_LF_diff_G = 0
        sum_LF_diff_B = 0
        
        diff_LF_R_list = []
        diff_LF_G_list = []
        diff_LF_B_list = []

        for i in range(num_minibatches_to_test):
            data = self.reader_test.next_minibatch(test_minibatch_size, input_map=test_input_map)

            pre_temp = data[self.input].asarray()[0]
            #print("Pre --------------------")
            #print(pre_temp)

            post_temp = data[self.label].asarray()[0]
            #print("Post --------------------")
            #print(post_temp)
            #print()

            pre_temp = data[self.input].asarray()[0]
            #print("Features --------------------")
            pre_temp[0] = [int(x) for x in pre_temp[0]]
            #print(pre_temp[0])

            post_temp = data[self.label].asarray()[0]
            #print("Labels -----------------------")
            post_temp[0] = [int(x) for x in post_temp[0]]
            #print(post_temp[0])
            label_feature_diff = abs(pre_temp[0] - post_temp[0])
            label_feature_diff = label_feature_diff.tolist()
            #with open(self.file_name, 'a') as outfile:
                    #outfile.write("Label-Feature Diff = "+ str(label_feature_diff) + "\n")

            prediction = self.z.eval(pre_temp[0])
            prediction_float = np.float32(prediction[0])
            # print(prediction_float)
            v = [int(round(elem)) for elem in prediction_float]
            #print("Predict -----------------------")
            #print(v)

            rest = abs(v - post_temp[0])
            #print("--------------------------------")
            #print(rest)
            rest = rest.tolist()
            #print(type(rest))
            #with open(self.file_name, 'a') as outfile:
                    #outfile.write("Label-Predict Diff = " + str(rest) + "\n")
            
            ''' 
            #percentage of diff
            for index, item in enumerate(rest):
                p = (item / post_temp[0][index]) * 100
                #print(str(round(p, 2)) + "%")
                with open(self.file_name, 'a') as outfile:
                    outfile.write(str(index+1) + ". " + str(round(p, 2)) + "%" + "\n")
            '''
            #print()
            #with open(self.file_name, 'a') as outfile:
                    #outfile.write("\n")

            if (rest[0] > max_R):
                max_R = rest[0]

            if (rest[1] > max_G):
                max_G = rest[1]

            if (rest[2] > max_B):
                max_B = rest[2]
                
                
            if (label_feature_diff[0] > max_LF_R):
                max_LF_R = label_feature_diff[0]

            if (label_feature_diff[1] > max_LF_G):
                max_LF_G = label_feature_diff[1]

            if (label_feature_diff[2] > max_LF_B):
                max_LF_B = label_feature_diff[2]
                
            sum_diff_R += rest[0]
            sum_diff_G += rest[1]
            sum_diff_B += rest[2]
            
            sum_LF_diff_R += label_feature_diff[0]
            sum_LF_diff_G += label_feature_diff[1]
            sum_LF_diff_B += label_feature_diff[2]
            
            diff_R_list.append(rest[0])
            diff_G_list.append(rest[1])
            diff_B_list.append(rest[2])
            
            diff_LF_R_list.append(label_feature_diff[0])
            diff_LF_G_list.append(label_feature_diff[1])
            diff_LF_B_list.append(label_feature_diff[2])

            
        with open(self.file_name, 'a') as outfile:
            outfile.write("max Label-Feature diff (R G B) : " + str(max_LF_R) + " " + str(max_LF_G) + " " + str(max_LF_B) + " \n")
            
        #print("max diff R G B : ", max_R, max_G, max_B)
        with open(self.file_name, 'a') as outfile:
            outfile.write("max Label-Predict diff (R G B) : " + str(max_R) + " " + str(max_G) + " " + str(max_B) + " \n")
            
        avg_LF_diff_R = sum_LF_diff_R / self.num_test_samples
        avg_LF_diff_G = sum_LF_diff_G / self.num_test_samples
        avg_LF_diff_B = sum_LF_diff_B / self.num_test_samples
        
        avg_LF_diff_R = round(avg_LF_diff_R, 4)
        avg_LF_diff_G = round(avg_LF_diff_G, 4)
        avg_LF_diff_B = round(avg_LF_diff_B, 4)
        
        avg_diff_R = sum_diff_R / self.num_test_samples
        avg_diff_G = sum_diff_G / self.num_test_samples
        avg_diff_B = sum_diff_B / self.num_test_samples
        
        avg_diff_R = round(avg_diff_R, 4)
        avg_diff_G = round(avg_diff_G, 4)
        avg_diff_B = round(avg_diff_B, 4)
        
        with open(self.file_name, 'a') as outfile:
            outfile.write("avg Label-Feature diff (R G B) : " + str(avg_LF_diff_R) + " " + str(avg_LF_diff_G) + " " + str(avg_LF_diff_B) + " \n")
        
        #print("avg diff R G B : ", avg_diff_R, avg_diff_G, avg_diff_B)
        with open(self.file_name, 'a') as outfile:
            outfile.write("avg Label-Predict diff (R G B) : " + str(avg_diff_R) + " " + str(avg_diff_G) + " " + str(avg_diff_B) + " \n")
            
        R_sd = self.cal_standard_deviation(diff_R_list, avg_diff_R)
        G_sd = self.cal_standard_deviation(diff_G_list, avg_diff_G)
        B_sd = self.cal_standard_deviation(diff_B_list, avg_diff_B)
        
        R_sd = round(R_sd, 4)
        G_sd = round(G_sd, 4)
        B_sd = round(B_sd, 4)
        
        #print("Standard Deviation diff R G B : ", R_sd ,G_sd ,B_sd)
        with open(self.file_name, 'a') as outfile:
            outfile.write("Standard Deviation diff (R G B) : " + str(R_sd) + " " + str(G_sd) + " " + str(B_sd) + "\n")

    def save_model(self, path):
        self.z.save(path)
        
    def cal_standard_deviation(self, diff_list, avg):
        summ = 0
        for i in diff_list:
            t = (i - avg)*(i - avg)
            summ += t
        sd = math.sqrt(summ/len(diff_list))
        #print(len(diff_list))
        return sd

    def start(self, hidden_layers_dim, learning_rate, minibatch_size, num_train_samples_per_sweep, num_test_samples, result_file_name):
        self.file_reader()
        self.config_parameter(hidden_layers_dim, learning_rate, minibatch_size, num_train_samples_per_sweep, num_test_samples, result_file_name)
        self.run_trainer()
        self.run_tester()
