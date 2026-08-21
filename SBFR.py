import os
import time
import numpy as np
import pandas as pd

from sygp.MethodsConfig import Methodsconfig
from sygp.SYGPConfig import SYGPConfig
from sygp.PrepareDataset import PrepareDataset
from sygp.SYGPMethods import SYGPMethods
from Argumentssygp import *
import os

# Full path of the current file
Subject = os.path.splitext(os.path.basename(__file__))[0]
# Just the file name
file_name = os.path.basename(__file__)
print("File name:", file_name)
DATASETS_DIR = 'datasetbreak2/'
cross_rate = 0.75
cross_over_type = 0
method = ['ML_Regression']
print(method)
functions = "simple"
# functions = "complex"

methods_config = Methodsconfig(method[0])
my_list = methods_config.get_parameter()
sygp = my_list[0]
classification = my_list[1]
Random_pop = my_list[2]
ML = my_list[3]
tor1 = my_list[4]
cross = my_list[5]
k_split_trainingset = my_list[6]
Normalization = my_list[7]
datasetName = my_list[8]
methodNames = my_list[9]
pop_size = my_list[10]
OUTPUT_DIR = my_list[11]
FITNESS_TYPE = my_list[12]

########################################################

sygp_config = SYGPConfig(
    sygp, classification, ML, Random_pop)

PrepDataset = PrepareDataset(
    sygp, classification, ML, Random_pop, Normalization, cross)

base1 = 1
base2 = 1
niche_size = 300

sygpmethods = SYGPMethods(
    sygp, FITNESS_TYPE, classification,
    Random_pop, ML, tor1, cross,k_split_trainingset, base1, base2
)

hirarchy = 0
usecorr = 0
method_=""
for which in datasetName:
    print('dataset:' + which)

    for method in methodNames:
        result_test = []
        result_train = []

        result_fitness = []
        fitness_end = []
        train_end = []

        test_end = []
        dim = []

        mae_train = []
        mae_test = []
        r2_train = []
        r2_test = []

        result_eval = []
        accuracy = []
        runtime_list = []
        modelName = sygp_config.get_model_name(method)
        sygpmethods.set_method_name(modelName)
        start = 0
        end = 9
        RUNS = range(start, end + 1)
        model_name = ""

        for r in RUNS:
            run_start = time.time()  # start timer

            X_train, y_train, X_test, y_test, X_val, y_val = PrepDataset.get_dataset(DATASETS_DIR, which, r)
            X_train = pd.concat([X_train, X_val], axis=0)
            y_train = pd.concat([y_train, y_val], axis=0)
            X_val, y_val = [], []
            sygpmethods.set_data_set(X_train, y_train, X_test, y_test, X_val, y_val)
            Num_class = len(np.unique((y_train)))

            if ML:
                method_name = method
                (train_end, test_end, mae_train, mae_test, r2_train, r2_test, method_, dim) = (sygpmethods.call_ML_Method
                    (method, train_end, test_end, mae_train, mae_test, r2_train, r2_test, r, which, dim))


            # compute runtime
            run_end = time.time()
            runtime = run_end - run_start
            print(f"Run {r} finished in {runtime:.2f} seconds")
            runtime_list.append(runtime)
            # save results immediately after each run

            PrepDataset.save_result(Subject,
                OUTPUT_DIR, which, dim, test_end, train_end, result_eval, result_fitness, fitness_end, result_train,
                result_test, modelName,
                base1, base2, MAX_NICHE_COUNT,
                functions, mae_train, mae_test,
                       r2_train, r2_test,method_,
                runtime=runtime_list  # <-- NEW
            )
