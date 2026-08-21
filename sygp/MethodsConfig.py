class Methodsconfig:
    sygp = False
    classification = False
    ML = False
    k_split_trainingset = False
    Normalization = False
    tor1 = False
    cross = 0
    pop_size = 200
    Random_pop = False

    def __init__(self, method1):
        self.sygp = False
        self.ML = False
        self.k_split_trainingset = False
        self.Normalization = False
        self.tor1 = False
        self.cross = 7
        self.pop_size = 200
        self.Random_pop = False
        self.FITNESS_TYPE = ['Accuracy']

        if method1 == 'ML_Regression':
            self.OUTPUT_DIR = 'ML_Regression/'
            self.ML = True
            self.methodNames = ['rand_pop_regression']
            self.datasetName = ['Concrete']
            self.FITNESS_TYPE = ["MSE"]

    def get_parameter(self):

        self.OUTPUT_DIR = "all_results/" + self.OUTPUT_DIR
        my_list = [self.sygp, self.classification, self.Random_pop, self.ML, self.tor1, self.cross,
                   self.k_split_trainingset, self.Normalization, self.datasetName, self.methodNames, self.pop_size, self.OUTPUT_DIR, self.FITNESS_TYPE]
        return my_list
