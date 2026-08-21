from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from .MahalanobisDistanceClassifier import MahalanobisDistanceClassifier
from .MulticlassLinearReg import MulticlassLinearReg
from .SimpleSum import SimpleSum
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import hinge_loss, mean_squared_error, log_loss
import pandas as pd
from .Individual import Individual
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, \
    HistGradientBoostingClassifier, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression


class MostSynergisticFeatures:
    training_X = None
    training_Y = None
    operators = None
    terminals = None
    max_depth = None
    dimensions = None
    size = 0
    depth = 0
    trainingPredictions = None
    testPredictions = None
    dataset_name = None
    validationPredictions = None
    fitness = None
    model = None
    kfold = None
    regressor = False
    model1 = None
    model2 = None
    model3 = None

    def __init__(self, max_depth, model_name="LogisticRegression", validation=True, kfold=5,
                 tr_x=None, tr_y=None, fitnessType="Accuracy", dataset_name="", classification=True):
        self.trainingAccuracy = None
        self.testAccuracy = None
        self.max_depth = max_depth
        self.model_name = model_name
        self.fitnessType = fitnessType
        self.dataset_name = dataset_name
        self.validation = validation
        self.training_X = tr_x
        self.training_Y = tr_y
        self.classification = classification
        self.kfold = kfold
        self.residual = pd.Series(self.training_Y).values if tr_y is not None else None
        self.model = self.createModel()
        self.kf = KFold(n_splits=self.kfold)
        if self.model_name in ('Ridge', 'DecisionTreeRegressor', 'RFreg', 'ExtraTreesRegressor',
                               'AdaBoostRegressor', 'GBDT', 'DART', 'XGBoost', 'LightGBM', 'CatBoost',
                               "RandomForestRegressor", "LinearRegression", "SimpleSum", "SVR",
                               "KNeighborsRegressor"):
            self.regressor = True

    # ---------------------- Dimension handling ----------------------
    def create(self, MSF, n_dims=1):
        self.dimensions = []
        for i in range(n_dims):
            n = MSF[i].dimensions[0]
            self.dimensions.append(n)

    def copy(self, dim):
        self.dimensions = dim

    def __gt__(self, other):
        sf = self.getFitness()
        sd = self.getNumberOfDimensions()
        ss = self.getSize()
        of = other.getFitness()
        od = other.getNumberOfDimensions()
        os = other.getSize()
        return (sf > of) or \
            (sf == of and sd < od) or \
            (sf == of and sd == od and ss < os)

    def __ge__(self, other):
        return self.getFitness() >= other.getFitness()

    def __str__(self):
        return ",".join([str(d) for d in self.dimensions])

    # ---------------------- Model creation ----------------------
    def createModel(self):
        # Check for LDA feasibility
        if self.model_name == "LDA":
            if self.training_X is not None and self.training_X.shape[0] > self.training_X.shape[1]:
                return LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
            else:
                print("Warning: Too few samples for LDA, using LogisticRegression instead.")
                return LogisticRegression(max_iter=1000)

        models = {
            "LinearRegression": LinearRegression(),
            "MahalanobisDistanceClassifier": MahalanobisDistanceClassifier(),
            "RandomForestClassifier": RandomForestClassifier(max_depth=5, random_state=42),
            "RS": BaggingClassifier(bootstrap=False, max_features=0.5),
            "DecisionTree": tree.DecisionTreeClassifier(random_state=42, max_depth=6),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42, max_depth=6),
            "Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Linear SVM": SVC(kernel="linear", C=1, max_iter=1000),
            "RBF SVM": SVC(gamma=2, C=1),
            "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
            "Neural Net": MLPClassifier(alpha=1, max_iter=1000),
            "AdaBoost": AdaBoostClassifier(),
            "Naive Bayes": GaussianNB(),
            "QDA": QuadraticDiscriminantAnalysis(),
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "1NN": KNeighborsClassifier(n_neighbors=1),
            "xgb": HistGradientBoostingClassifier(min_samples_leaf=10, max_iter=50),
            "SimpleSum": SimpleSum(),
            "SVR": SVR(),
            "KNeighborsRegressor": neighbors.KNeighborsRegressor(),
            "MLPRegressor": MLPRegressor(random_state=1, max_iter=500),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=50, random_state=0, oob_score=True),
            "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=200, n_jobs=-1),
            "AdaBoostRegressor": AdaBoostRegressor(n_estimators=200),
            "GBDT": GradientBoostingRegressor(n_estimators=200),
            "DART": LGBMRegressor(n_jobs=1, n_estimators=200, boosting_type='dart', xgboost_dart_mode=True),
            "XGBoost": XGBRegressor(n_jobs=1, n_estimators=200),
            "LightGBM": LGBMRegressor(n_jobs=1, n_estimators=200),
            "CatBoost": CatBoostRegressor(n_estimators=200, thread_count=1, verbose=False, allow_writing_files=False),
            "Ridge": Ridge(),
            "MulticlassLinearReg": MulticlassLinearReg()
        }
        return models.get(self.model_name, LogisticRegression(max_iter=1000))

    # ---------------------- Fit ----------------------
    def fit(self, Tr_x, Tr_y):
        try:
            self.model = self.createModel()
            hyper_X = self.convert(Tr_x)
            hyper_X = hyper_X.fillna(0)
            if isinstance(Tr_y, pd.Series) or isinstance(Tr_y, pd.DataFrame):
                Tr_y = Tr_y.fillna(0)
            if self.model_name != "LDA" or hyper_X.shape[0] > hyper_X.shape[1]:
                self.model.fit(hyper_X, Tr_y)
            else:
                print("Warning: Skipping LDA fit due to too few samples.")
        except ValueError as ve:
            print(f"ValueError during model fitting: {ve}")
        except Exception as e:
            print(f"Unexpected error during model fitting: {e}")

    # ---------------------- Dimension/Residual utilities ----------------------
    def getSize(self):
        if not self.size:
            self.size = sum(n.getSize() for n in self.dimensions)
        return self.size

    def getDepth(self):
        if not self.depth:
            self.depth = max([dimension.getDepth() for dimension in self.dimensions])
        return self.depth

    def getResidual(self, Tr_x, residual):
        hyper_x = self.convert(Tr_x)
        M = self.model
        M.fit(hyper_x, residual)
        if self.model_name in ["LDA", "LogisticRegression", "MulticlassLinearReg"]:
            predict = M.predict_proba(hyper_x)[:, 1]
        else:
            predict = M.predict(hyper_x)
        residual = residual - predict
        return residual

    def getDimensions(self):
        return [dim.clone() for dim in self.dimensions]

    def getNumberOfDimensions(self):
        return len(self.dimensions)

    # ---------------------- Convert and predict ----------------------
    def convert(self, X):
        ret = pd.DataFrame()
        for i in range(len(self.dimensions)):
            ret["#" + str(i)] = self.dimensions[i].calculate(X)
        return ret

    def predict(self, X):
        hyper_X = self.convert(X)
        try:
            predictions = self.model.predict(hyper_X)
        except Exception as e:
            print(f"Predict failed: {e}")
            predictions = np.zeros(len(X))
        return predictions

    def predict_proba(self, X):
        hyper_X = self.convert(X)
        try:
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba(hyper_X)
            else:
                pred = self.predict(X)
                return np.vstack([1 - pred, pred]).T
        except Exception as e:
            print(f"Predict_proba failed: {e}")
            return np.zeros((len(X), 2))

    def revise_output(self, y):
        y = np.array(y)
        y[y > 0] = 1
        y[y <= 0] = -1
        return y

    # ---------------------- Fitness and evaluation ----------------------
    def get_mse(self, y_pred, y_true):
        return np.sqrt(mean_squared_error(y_pred, y_true))

    def getFitness(self, tr_x, tr_y):
        try:
            self.fit(tr_x, tr_y)
            self.getTrainingPredictions(tr_x)
            if self.fitnessType in ["Accuracy", "LDA"]:
                if self.regressor:
                    self.trainingPredictions = self.revise_output(self.trainingPredictions)
                return accuracy_score(self.trainingPredictions, tr_y)
            elif self.fitnessType == "CrossEntropy":
                return -log_loss(tr_y, self.trainingPredictions)
            elif self.fitnessType == "MSE":
                return -self.get_mse(self.trainingPredictions, tr_y)
        except Exception as e:
            print(f"getFitness error: {e}")
            return -1e6

    def getFitnesses(self, tr_x=None, tr_y=None):
        if tr_x is None:
            tr_x = self.training_X
        if tr_y is None:
            tr_y = self.training_Y
        if self.validation:
            self.fitness = self.threefoldcross_validation(tr_x, tr_y)
        else:
            self.fitness = self.getFitness(tr_x, tr_y)
        return self.fitness

    def getTrainingPredictions(self, training_X):
        self.trainingPredictions = self.predict(training_X)
        return self.trainingPredictions

    def getTestPredictions(self, X):
        self.testPredictions = self.predict(X)
        return self.testPredictions

    # ---------------------- Three-fold cross-validation ----------------------
    def create_k_fol_data(self, tr_X, tr_y):
        k = 3
        hyper_X = self.convert(tr_X)
        n = len(hyper_X)
        splits = [0, n // 3, 2 * n // 3, n]
        X1, X2, X3 = hyper_X.iloc[splits[0]:splits[1]], hyper_X.iloc[splits[1]:splits[2]], hyper_X.iloc[
                                                                                           splits[2]:splits[3]]
        Y1, Y2, Y3 = tr_y[splits[0]:splits[1]], tr_y[splits[1]:splits[2]], tr_y[splits[2]:splits[3]]
        Y12 = pd.concat([Y1, Y2])
        Y13 = pd.concat([Y1, Y3])
        Y23 = pd.concat([Y2, Y3])
        return X1, X2, X3, Y12, Y13, Y23, Y1, Y2, Y3

    def threefoldcross_validation(self, tr_X, tr_y):
        try:
            X1, X2, X3, Y12, Y13, Y23, Y1, Y2, Y3 = self.create_k_fol_data(tr_X, tr_y)
            M1 = self.model
            M2 = self.model
            M3 = self.model

            M1.fit(pd.concat([X1, X2]), Y12)
            P1 = M1.predict(X3)

            M2.fit(pd.concat([X1, X3]), Y13)
            P2 = M2.predict(X2)

            M3.fit(pd.concat([X2, X3]), Y23)
            P3 = M3.predict(X1)

            if self.fitnessType == "MSE":
                f1 = -self.get_mse(P1, Y3)
                f2 = -self.get_mse(P2, Y2)
                f3 = -self.get_mse(P3, Y1)
            else:
                f1 = accuracy_score(P1, Y3)
                f2 = accuracy_score(P2, Y2)
                f3 = accuracy_score(P3, Y1)
            return (f1 + f2 + f3) / 3
        except Exception as e:
            print(f"[threefoldcross_validation] Error: {e}")
            return -1e6

    # ---------------------- Remaining metrics ----------------------
    def getMSE(self, X, Y, pred=None):
        if pred == "Tr":
            pred = self.getTrainingPredictions(X)
        elif pred == "Te":
            pred = self.getTestPredictions(X)
        else:
            pred = self.predict(X)
        return -mean_squared_error(pred, Y)

    def getAccuracy(self, X, Y, pred=None):
        if pred == "Tr":
            pred = self.getTrainingPredictions(X)
        elif pred == "Te":
            pred = self.getTestPredictions(X)
        else:
            pred = self.predict(X)
        return accuracy_score(pred, Y)

    def getWaF(self, X, Y, pred=None):
        if pred == "Tr":
            pred = self.getTrainingPredictions(X)
        elif pred == "Te":
            pred = self.getTestPredictions(X)
        else:
            pred = self.predict(X)
        return f1_score(pred, Y, average="weighted")

    def getKappa(self, X, Y, pred=None):
        if pred == "Tr":
            pred = self.getTrainingPredictions(X)
        elif pred == "Te":
            pred = self.getTestPredictions(X)
        else:
            pred = self.predict(X)
        return cohen_kappa_score(pred, Y)

    # ---------------------- Synergistic features utilities ----------------------
    def calculate(self, sample):
        return [self.dimensions[i].calculate(sample) for i in range(len(self.dimensions))]

    def prun(self, min_dim=1, simp=False):
        dup = self.dimensions[:]
        i = 0
        ind = Individual(self.operators, self.terminals, self.max_depth, self.model_name, self.fitnessType)
        ind.copy(dup)
        ind.fit(self.training_X, self.training_Y)

        while i < len(dup) and len(dup) > min_dim:
            dup2 = dup[:]
            dup2.pop(i)
            ind2 = Individual(self.operators, self.terminals, self.max_depth, self.model_name, self.fitnessType)
            ind2.copy(dup2)
            ind2.fit(self.training_X, self.training_Y)

            if ind2 >= ind:
                ind = ind2
                dup = dup2
                i -= 1
            i += 1

        self.dimensions = dup
        self.trainingAccuracy = None
        self.testAccuracy = None
        self.size = None
        self.depth = None
        self.model = None
        self.fit(self.training_X, self.training_Y)

        if simp:
            for d in self.dimensions:
                done = False
                while not done:
                    state = str(d)
                    d.prun(self.training_X)
                    done = state == str(d)

    def cross_entropy_loss(self, Y, pred):
        return -log_loss(Y, pred)

    def getTrainingMeasure(self, tr_X, tr_Y):
        """
        Returns the training measure (fitness) of the current model.
        Uses accuracy for classification and negative MSE for regression.
        """
        try:
            self.fit(tr_X, tr_Y)
            self.getTrainingPredictions(tr_X)

            if self.classification:
                if self.regressor:
                    self.trainingPredictions = self.revise_output(self.trainingPredictions)
                return accuracy_score(self.trainingPredictions, tr_Y)
            else:
                return -self.get_mse(self.trainingPredictions, tr_Y)

        except Exception as e:
            print(f"[getTrainingMeasure] Error: {e}")
            return -1e6

    def getTestMeasure(self, test_X, test_Y):
        """
        Returns the test measure (fitness) of the current model.
        Uses accuracy for classification and negative MSE for regression.
        """
        try:
            self.getTestPredictions(test_X)
            if self.classification:
                if self.regressor:
                    self.testPredictions = self.revise_output(self.testPredictions)
                return accuracy_score(self.testPredictions, test_Y)
            else:
                return -self.get_mse(self.testPredictions, test_Y)
        except Exception as e:
            print(f"[getTestMeasure] Error: {e}")
            return -1e6
