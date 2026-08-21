from sklearn.preprocessing import StandardScaler
from Argumentssygp import *
import os
import csv
import numpy as np
import pandas as pd


class PrepareDataset:
    use_residual = False
    classification = True
    ML = False
    one_pop = False
    Random_pop = False
    sygp = False
    Normalization = True

    def __init__(self, sygp, classification, ml, random_pop, normalization, cross=7):
        self.sygp = sygp
        self.classification = classification
        self.ML = ml
        self.Random_pop = random_pop
        self.Normalization = normalization
        self.cross = cross

    def masking(self, y, k):
        y_prim = y.copy()
        for index in range(len(y)):
            if y_prim[index] == k:
                y_prim[index] = 1
            else:
                y_prim[index] = -1
        return y_prim

    def get_dataset(self, DATASETS_DIR, which, r):
        df_X = pd.read_csv(DATASETS_DIR + which + str(r) + 'train.csv')
        x_train = df_X.iloc[:, :-1]
        y_train = df_X['Y']

        df_Y = pd.read_csv(DATASETS_DIR + which + str(r) + 'test.csv')
        x_test = df_Y.iloc[:, :-1]
        y_test = df_Y['Y']

        df_Y = pd.read_csv(DATASETS_DIR + which + str(r) + 'val.csv')
        x_val = df_Y.iloc[:, :-1]
        y_val = df_Y['Y']
        if self.Normalization:
            # scaler = MinMaxScaler()
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
            x_val_scaled = scaler.transform(x_val)
            x_train = pd.DataFrame(x_train_scaled, columns=x_train.columns)
            x_test = pd.DataFrame(x_test_scaled, columns=x_test.columns)
            x_val = pd.DataFrame(x_val_scaled, columns=x_test.columns)

        return x_train, y_train, x_test, y_test, x_val, y_val

    def save_result(self, subject, OUTPUT_DIR1, which, dim, test_end, train_end, result_eval,
                    result_fitness, fitness_end, result_train,
                    result_test, method, base1, base2, MAX_NICHE_COUNT, functions,
                    mae_train=None, mae_test=None, r2_train=None, r2_test=None, method_=None, runtime=None):

        """
        Save all results and metrics (including runtime) into CSV files.
        """
        method = method + '_' + method_

        # =============================
        # Determine output directory and filename
        # =============================
        if self.sygp:
            OUTPUT_DIR = os.path.join(OUTPUT_DIR1, subject, str(which))
            outputfile = f'_{which}{MAX_Evaluation}'
            end_of_path = (f"{method}pop{POPULATION_SIZE}_type_{self.cross}{outputfile}"
                           f"{FITNESS_TYPE[0]}{VALIDATION}{base1}{base2}{MAX_NICHE_COUNT}{functions}")
        else:
            OUTPUT_DIR = OUTPUT_DIR1
            outputfile = f'_{which}'
            end_of_path = f"{method}{outputfile}"

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # =============================
        # Helper function
        # =============================
        def _save_csv(filename, data):
            if data is None:
                return
            path = os.path.join(OUTPUT_DIR, f"{filename}{end_of_path}.csv")
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                if isinstance(data, (list, tuple, np.ndarray)):
                    writer.writerow(data)
                else:
                    writer.writerow([data])

        # =============================
        # Always save core results
        # =============================
        _save_csv("test", test_end)
        _save_csv("train", train_end)
        _save_csv("dim", dim)
        if self.sygp:
            _save_csv("dim", dim)
            _save_csv("result_eval", result_eval)
            _save_csv("result_fitness", result_fitness)
            _save_csv("fitness_end", fitness_end)
            _save_csv("result_train", result_train)
            _save_csv("result_test", result_test)

        # =============================
        # Save classification or regression metrics conditionally
        # =============================

        # 🔹 Regression metrics
        if mae_train is not None: _save_csv("mae_train", mae_train)
        if mae_test is not None:  _save_csv("mae_test", mae_test)
        if r2_train is not None:  _save_csv("r2_train", r2_train)
        if r2_test is not None:   _save_csv("r2_test", r2_test)

        # =============================
        # Save runtime if available
        # =============================
        if runtime is not None:
            if not isinstance(runtime, (list, tuple, np.ndarray)):
                runtime = [runtime]
            _save_csv("runtime", runtime)

        print(f"✅ Results saved to: {OUTPUT_DIR}")
