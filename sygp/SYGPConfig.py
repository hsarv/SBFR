
class SYGPConfig:
    use_residual = False
    classification = True
    gsgp_standard = False
    ML = False
    ML_default = False
    one_pop = False
    Random_pop = False
    sygp = False
    gsgp = False

    def __init__(self, sygp, classification, ML, Random_pop):
        self.sygp = sygp
        self.classification = classification
        self.ML = ML
        self.Random_pop = Random_pop

    def get_model_name(self, method):
        modelName = ''
        if method == 'KNN':
            modelName = "Nearest Neighbors"
        elif method == 'SimpleSum':
            modelName = "SimpleSum"
        elif method == 'LR':
            modelName = "LogisticRegression"
        elif method == '1NN':
            modelName = "1NN"
        elif method == 'DT':
            modelName = "DecisionTree"
        elif method == 'RF':
            modelName = "RandomForestClassifier"
        elif method == 'MLP':
            modelName = "Neural Net"
        elif method == 'NC':
            modelName = "MahalanobisDistanceClassifier"
        elif method == 'SVM':
            modelName = "Linear SVM"
        elif method == 'RS':
            modelName = "RS"
        elif method == 'LDA':
            modelName = "LDA"
        elif method == 'XGB':
            modelName = "xgb"
        elif method == "Lir":
            modelName = "LinearRegression"
        elif method == "LDA":
            modelName = "LDA"
        elif method == "RFreg":
            modelName = "RFreg"
        else:
            modelName = method
        return modelName

