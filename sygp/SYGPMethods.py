from gplearn.genetic import SymbolicRegressor
from sygp.SYGP import SYGP
from Argumentssygp import *
import numpy as np
from sklearn import tree, neighbors
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
    GradientBoostingRegressor)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score)
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


class SYGPMethods:
    classification = True
    ML = False
    one_pop = False
    Random_pop = False
    sygp_flag = False
    modelName = ""
    methodName = ""

    def root_mean_squared_error(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def __init__(self, sygp_flag, FITNESS_TYPE,
                 classification, random_pop, ml, tor1, cross, k_split_training_set, base1, base2):
        self.y_val = None
        self.y_test = None
        self.X_val = None
        self.X_test = None
        self.X_train = None
        self.y_train = None
        self.sygp_flag = sygp_flag
        self.classification = classification
        self.ML = ml
        self.stgp_cross = cross
        self.tor1 = tor1
        self.sygp_OnePopSetFitness = True
        self.sygp_standardAproach = False
        self.sygp_Random_pop = random_pop
        self.FITNESS_TYPE = FITNESS_TYPE
        self.sygp_k_split_trainingset = k_split_training_set
        self.SYGP_MultiPop_layer = True
        self.sygp_allPopulation = []
        self.sygp_population = []
        self.sygp_MSF = []
        self.sygp_initial_random = 1
        self.cross = cross
        self.base1 = base1
        self.base2 = base2

    def _calculate_regression_metrics(self, y_true, y_pred):
        metrics = {'rmse': np.sqrt(mean_squared_error(y_true, y_pred)), 'mae': mean_absolute_error(y_true, y_pred),
                   'r2': r2_score(y_true, y_pred)}
        return metrics

    def call_ML_Method(self, method, train_end, test_end,
                       mae_train=None, mae_test=None,
                       r2_train=None, r2_test=None, r=None, which=None, dim=None):

        X_train, y_train = self.X_train, self.y_train
        X_test, y_test = self.X_test, self.y_test

        self.modelName = ""
        clf = None
        if method == "Lir":
            clf = LinearRegression()
        elif method == 'SVR':
            clf = SVR()
        elif method == 'KNeighborsRegressor':
            clf = neighbors.KNeighborsRegressor()
        elif method == 'MLPRegressor':
            clf = MLPRegressor(random_state=1, max_iter=500)
        elif method == 'DecisionTreeRegressor':
            clf = tree.DecisionTreeRegressor()
        elif method == 'RandomForestRegressor':
            clf = RandomForestRegressor(n_estimators=50, random_state=0, oob_score=True)
        elif method == 'ExtraTreesRegressor':
            clf = ExtraTreesRegressor(n_estimators=200, n_jobs=-1)
        elif method == 'AdaBoostRegressor':
            clf = AdaBoostRegressor(n_estimators=200)
        elif method == 'GBDT':
            clf = GradientBoostingRegressor(n_estimators=200)
        elif method == 'LightGBM':
            clf = LGBMRegressor(n_jobs=1, n_estimators=200)
        elif method == 'XGBoost':
            clf = XGBRegressor(n_jobs=1, n_estimators=200)
        elif method == 'CatBoost':
            clf = CatBoostRegressor(n_estimators=200, thread_count=1, verbose=False, allow_writing_files=False)
        elif method == 'Ridge':
            clf = Ridge()
        elif method == 'GPlearn':
            clf = SymbolicRegressor(population_size=5000,
                                    generations=30, stopping_criteria=0.01,
                                    p_crossover=0.7, p_subtree_mutation=0.1,
                                    p_hoist_mutation=0.05, p_point_mutation=0.1,
                                    max_samples=0.9, verbose=1,
                                    parsimony_coefficient=0.01, random_state=r)

        elif method == 'rand_pop_regression':
            self.sygp_allPopulation = []
            self.sygp_population = []
            self.sygp_MSF = []
            self.sygp_initial_random = 1

            self.methodName = "SYGP_RandomPop_regression_sta" + self.modelName
            self.modelName = 'SVR'
            # ['SVR','DecisionTreeRegressor','RandomForestRegressor','Ridge','KNeighborsRegressor','MLPRegressor']
            pop_size = 20
            niche_size = 300
            MAX_NICHE_COUNT2 = 30

            clf = SYGP(self.sygp_allPopulation, self.sygp_population, self.sygp_MSF, OPERATORS, MAX_DEPTH,
                       pop_size, MAX_GENERATION, MAX_Evaluation, LIMIT_DEPTH,
                       DIM_MIN, DIM_MAX, THREADS, r,
                       KFOLD, VERBOSE,
                       self.modelName, self.FITNESS_TYPE[0], which, MAX_NICHE_COUNT2,
                       self.methodName, self.classification,
                       self.sygp_k_split_trainingset, self.base1, self.base2)

        if method == 'rand_pop_regression':
            clf.fit(X_train, y_train, self.X_test, self.y_test)
        else:
            clf.fit(X_train, y_train)

        if method == 'rand_pop_regression':
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)

            best_program = clf.msfvector

            for ii in range(len(best_program.dimensions)):
                node = best_program.dimensions[ii]
                print(node.getSize())
                node.prun(self.X_train)
                print(node.getSize())
                node.visualize_tree(node,
                                    'expression' + 'indiv' + str(ii) + 'run' + str(
                                        r) + which + self.methodName + '.png')
            dim += [clf.msfvector.getNumberOfDimensions()]
        else:
            y_pred_train = clf.predict(X_train)
            y_pred_test = clf.predict(X_test)

        train_metrics = self._calculate_regression_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_regression_metrics(y_test, y_pred_test)

        train_end += [train_metrics['rmse']]
        test_end += [test_metrics['rmse']]

        if mae_train is not None: mae_train += [train_metrics['mae']]
        if mae_test is not None: mae_test += [test_metrics['mae']]
        if r2_train is not None: r2_train += [train_metrics['r2']]
        if r2_test is not None: r2_test += [test_metrics['r2']]

        return (train_end, test_end,
                mae_train, mae_test,
                r2_train, r2_test, method + self.modelName, dim)

    def set_data_set(self, x_train, y_train, x_test, y_test, x_val, y_val):
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test
        self.X_val = x_val
        self.y_val = y_val

    def set_method_name(self, model_name):
        self.modelName = model_name
