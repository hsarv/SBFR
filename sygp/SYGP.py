import numpy as np
from .Individual import Individual
from .MostSynergisticFeatures import MostSynergisticFeatures
from random import Random


class ClassifierNotTrainedError(Exception):
    """ You tried to use the classifier before training it. """


def __init__(self, expression, message=""):
    self.expression = expression
    self.message = message


class SYGP:
    classification = None
    operators = None
    method_name = None
    max_initial_depth = None
    population_size = None
    threads = None
    random_state = 42
    rng = None
    max_depth = None
    max_generation = None
    max_evaluation = None
    num_evaluation = 0
    tournament_size = None
    elitism_size = None
    dim_min = None
    dim_max = None
    model_name = None
    dataset_name = None
    fitnessType = None
    verbose = None
    this_eval = 0
    correlationType = None
    crossover_type = None
    max_niche_count = None
    terminals = None
    population = None
    allPopulation = None
    pool_population = None
    currentGeneration = 1
    random_pop = None
    standardAproach = None
    OnePop = None
    OnePopSetFitness = None
    bestIndividual: Individual = None
    MSF = None
    msfvector = None
    fitnessOverTime = None
    validationCheck = None
    cross_rate = None

    # trainingAccuracyOverTime = None
    # testAccuracyOverTime = None
    # trainingWaFOverTime = None
    # testWaFOverTime = None
    # trainingKappaOverTime = None
    # testKappaOverTime = None
    # trainingMSEOverTime = None
    # testMSEOverTime = None
    # sizeOverTime = None
    # dimensionsOverTime = None
    # generationTimes = None

    def checkIfTrained(self):
        if self.population is None:
            raise ClassifierNotTrainedError(
                "The classifier must be trained using the fit(Tr_X, Tr_Y) method before being used.")

    def __init__(self, allPopulation, population, MSF, operators=[("+", 2), ("-", 2), ("*", 2), ("/", 2)],
                 max_initial_depth=6, population_size=1,
                 max_generation=100, max_evaluation=50000, max_depth=17,
                 dim_min=1, dim_max=9999, threads=1, random_state=42, kfold=5, verbose=True,
                 model_name="MahalanobisDistanceClassifier",
                 fitnessType="Accuracy", dataset="heart.csv", max_niche_count=30, method_name="",
                 classification=True, k_split_trainingset=False, base1=1, base2=1):
        self.evalOverTime = None
        self.testAccuracyOverTime = None
        self.dimOverTime = None
        self.trainingAccuracyOverTime = None
        self.validationDir = None
        self.Te_y = None
        self.validationvector = None
        self.Te_x = None
        self.V_y = None
        self.V_x = None
        self.Tr_y = None
        self.Tr_x = None
        self.generationTimes = None
        self.dimensionsOverTime = None
        self.trainingMSEOverTime = None
        self.sizeOverTime = None
        self.testWaFOverTime = None
        self.testMSEOverTime = None
        self.testKappaOverTime = None
        self.trainingWaFOverTime = None
        self.trainingKappaOverTime = None
        self.bestFitness = None
        self.validation = False
        self.method_name = method_name
        self.kfold = kfold
        self.classification = classification
        self.k_split_trainingset = k_split_trainingset
        if sum([0 if op in [("+", 2), ("-", 2), ("*", 2), ("/", 2)] else 0 for op in operators]) > 0:
            print("[Warning] Some of the following operators may not be supported:", operators)
        self.operators = operators
        self.max_initial_depth = max_initial_depth
        self.population_size = population_size
        self.threads = max(1, threads)
        self.random_state = random_state
        self.rng = Random(random_state)
        self.max_depth = max_depth
        self.max_generation = max_generation
        self.max_evaluation = max_evaluation

        self.dim_min = max(1, dim_min)
        self.dim_max = max(1, dim_max)
        self.model_name = model_name
        self.fitnessType = fitnessType
        self.verbose = verbose
        self.dataset_name = dataset
        self.max_niche_count = max_niche_count
        self.correlationType = "mi"

        self.allPopulation = allPopulation
        self.population = population
        self.MSF = MSF

        self.total_rate_cross = 0
        self.total_rate_mute = 0
        self.success_rate_cross = 0
        self.success_rate_mute = 0
        self.validationDecrease = False
        self.msf_temp = None
        self.validationCheck = False
        self.base1 = base1
        self.base2 = base2

    def __str__(self):
        self.checkIfTrained()
        return str(self.getBestIndividual())

    def getCurrentGeneration(self):
        return self.currentGeneration

    def getBestIndividual(self):
        self.checkIfTrained()
        return self.msfvector

    def getAccuracyOverTime(self):
        self.checkIfTrained()
        return [self.trainingAccuracyOverTime, self.testAccuracyOverTime]

    def getWaFOverTime(self):
        self.checkIfTrained()
        return [self.trainingWaFOverTime, self.testWaFOverTime]

    def getKappaOverTime(self):
        self.checkIfTrained()

        return [self.trainingKappaOverTime, self.testKappaOverTime]

    def getMSEOverTime(self):
        self.checkIfTrained()
        return [self.trainingMSEOverTime, self.testMSEOverTime]

    def getSizesOverTime(self):
        self.checkIfTrained()
        return [self.sizeOverTime, self.dimensionsOverTime]

    def getGenerationTimes(self):
        self.checkIfTrained()
        return self.generationTimes

    def predict(self, dataset):
        self.checkIfTrained()
        return "Population Not Trained" if self.msfvector is None else self.msfvector.predict(dataset)

    def stoppingCriteria(self, patience: int = 100):
        """
        Stop training if:
          - Max evaluations reached
          - Perfect training solution found
          - Validation error decreased
          - No fitness improvement for `patience` iterations
        """

        Stock = False
        perfectTraining = 0
        evalLimit = False

        # --- Stop if maximum evaluations reached
        if self.num_evaluation >= self.max_evaluation:
            evalLimit = True

        # --- Stop if perfect training is achieved
        if (
                self.msfvector.getFitnesses(self.Tr_x, self.Tr_y) == 1
                or self.msfvector.getTrainingMeasure(self.Tr_x, self.Tr_y) == 1):
            perfectTraining = 1
            print("perfectTraining")

        # --- Stop if no improvement in last `patience` iterations
        if len(self.fitnessOverTime) >= patience:
            recent = self.fitnessOverTime[-patience:]
            if len(np.unique(recent)) == 1:  # no change
                Stock = True
                print(f"Stopped: No fitness improvement in last {patience} iterations")

        # --- Stop if validation performance decreases
        if self.validationDecrease:
            print("validationDecrease")

        return perfectTraining or evalLimit or self.validationDecrease or Stock

    def creat_new_msf_vector(self, msf_temp, MSF, fitness=None):

        self.msfvector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
                                                 self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name,
                                                 self.classification)
        self.msfvector.create(MSF, len(MSF))
        neg = 0
        if self.validationCheck:
            self.msfvector.getTrainingMeasure(self.Tr_x, self.Tr_y)
            self.validationvector.append(self.msfvector.getTestMeasure(self.V_x, self.V_y))
            if (self.validationvector[-1] - self.validationvector[-2]) > 0:
                self.validationDir.append(1)
            elif (self.validationvector[-1] - self.validationvector[-2]) == 0:
                self.validationDir.append(0)
            else:
                self.validationDir.append(-1)
            # ------
            length_of_search = len(self.validationDir)
            for i in range(length_of_search):
                if self.validationDir[-1 - i] == 1:
                    break
                elif self.validationDir[-1 - i] == 0:
                    pass
                else:
                    neg = neg + 1
        if neg > 2:
            self.validationDecrease = True
            self.MSF = msf_temp.copy()
            self.msfvector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
                                                     self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name,
                                                     self.classification)
            self.msfvector.create(self.MSF, len(self.MSF))
            self.bestFitness = self.msfvector.getFitnesses(self.Tr_x, self.Tr_y)
            del self.validationDir[-1]
            del self.validationvector[-1]

        else:
            self.validationDecrease = False
            if fitness is not None:
                self.bestFitness = fitness
            # if not self.bestFitness ==self.msfvector.getFitnesses(self.Tr_x, self.Tr_y):
            # 	print('vay')
            else:
                self.bestFitness = self.msfvector.getFitnesses(self.Tr_x, self.Tr_y)

    def fit(self, Tr_x, Tr_y, Te_x=None, Te_y=None, V_x=None, V_y=None):
        if self.verbose:
            print("  > Parameters")
            print("    > SYGP Type:       " + str(self.method_name))
            print("    > Random State:       " + str(self.random_state))
            print("    > Random pop:       " + str(self.random_pop))
            print("    > Tournament Size:    " + str(self.tournament_size))
            print("    > Validation:         " + str(self.validation))
            print("    > CrossoverType:         " + str(self.crossover_type))
            print("    > Wrapped Model:      " + self.model_name)
            print("    > dataset_name:      " + self.dataset_name)
            print("    > K fold:            " + str(self.kfold))
            print("    > population size:            " + str(self.population_size))
            print("    > Fitness Type:            " + str(self.fitnessType))
            print("    > fitness:            " + str(self.OnePopSetFitness))

        self.Tr_x = Tr_x
        self.Tr_y = Tr_y
        self.Te_x = Te_x
        self.Te_y = Te_y
        self.V_x = V_x
        self.V_y = V_y
        self.pool_population = []
        self.validationvector = []
        self.terminals = list(Tr_x.columns)
        self.validationvector.append(0)
        self.validationDir = []

        ind = Individual(self.operators, self.terminals, self.max_depth, self.model_name, self.fitnessType)
        ind.create(self.rng, n_dims=self.dim_min)
        self.MSF.append(ind)
        self.creat_new_msf_vector(self.MSF, self.MSF)

        self.IndividualInsert(self.population_size)

        self.fitnessOverTime = []
        self.trainingAccuracyOverTime = []
        self.testAccuracyOverTime = []
        self.dimOverTime = []
        self.evalOverTime = []

        # if not self.Te_x is None:
        # 	self.trainingAccuracyOverTime = []
        # 	self.testAccuracyOverTime = []
        # 	self.trainingWaFOverTime = []
        # 	self.testWaFOverTime = []
        # 	self.trainingKappaOverTime = []
        # 	self.testKappaOverTime = []
        # 	self.trainingMSEOverTime = []
        # 	self.testMSEOverTime = []
        # 	self.sizeOverTime = []
        # 	self.dimensionsOverTime = []
        # 	self.generationTimes = []

        if self.verbose:
            print("  > Running log:")
        overtime = True
        while self.num_evaluation < self.max_evaluation:
            if not self.stoppingCriteria():
                self.next_generation()
            else:
                self.num_evaluation = self.max_evaluation
            if overtime:
                self.evalOverTime.append(self.num_evaluation)
                self.fitnessOverTime.append(self.msfvector.getFitnesses(self.Tr_x, self.Tr_y))
                self.trainingAccuracyOverTime.append(self.msfvector.getTrainingMeasure(self.Tr_x, self.Tr_y))
                self.testAccuracyOverTime.append(self.msfvector.getTestMeasure(self.Te_x, self.Te_y))

                self.dimOverTime.append(self.msfvector.getNumberOfDimensions())
                self.currentGeneration += 1
        if overtime:
            self.evalOverTime.append(self.num_evaluation)
            self.fitnessOverTime.append(self.msfvector.getFitnesses(self.Tr_x, self.Tr_y))
            self.trainingAccuracyOverTime.append(self.msfvector.getTrainingMeasure(self.Tr_x, self.Tr_y))
            self.testAccuracyOverTime.append(self.msfvector.getTestMeasure(self.Te_x, self.Te_y))
            self.dimOverTime.append(self.msfvector.getNumberOfDimensions())
            self.currentGeneration += 1

        # if not self.Te_x is None:
        # 	if self.fitnessType in ["Accuracy", "2FOLD", "WAF"]:
        # 		# self.trainingAccuracyOverTime.append(self.msfvector.getAccuracy(self.Tr_x, self.Tr_y, pred="Tr"))
        # 		# self.testAccuracyOverTime.append(self.msfvector.getAccuracy(self.Te_x, self.Te_y, pred="Te"))
        # 		# # self.trainingWaFOverTime.append(self.msfvector.getWaF(self.Tr_x, self.Tr_y, pred="Tr"))
        # 		# self.testWaFOverTime.append(self.msfvector.getWaF(self.Te_x, self.Te_y, pred="Te"))
        # 		# self.trainingKappaOverTime.append(self.msfvector.getKappa(self.Tr_x, self.Tr_y, pred="Tr"))
        # 		# self.testKappaOverTime.append(self.msfvector.getKappa(self.Te_x, self.Te_y, pred="Te"))
        # 		# self.trainingMSEOverTime.append(0)
        # 		# self.testMSEOverTime.append(0)
        # 	elif self.fitnessType in ["MSE"]:
        # 		self.trainingAccuracyOverTime.append(0)
        # 		self.testAccuracyOverTime.append(0)
        # 		self.trainingWaFOverTime.append(0)
        # 		self.testWaFOverTime.append(0)
        # 		self.trainingKappaOverTime.append(0)
        # 		self.testKappaOverTime.append(0)
        # 		self.trainingMSEOverTime.append(self.msfvector.getMSE(self.Tr_x, self.Tr_y, pred="Tr"))
        # 		self.testMSEOverTime.append(self.msfvector.getMSE(self.Te_x, self.Te_y, pred="Te"))

    # 		self.sizeOverTime.append(self.msfvector.getSize())
    # 		self.dimensionsOverTime.append(self.msfvector.getNumberOfDimensions())
    # 		self.generationTimes.append(duration)
    # # prun the final individual
    # self.getBestIndividual().prun(min_dim = self.dim_min, simp=True)

    def IndividualInsert(self, pop_size):
        for i in range(pop_size):
            offspring = Individual(self.operators, self.terminals, self.max_depth, self.model_name, self.fitnessType)
            offspring.create(self.rng, n_dims=self.dim_min)
            self.updatePopulation(offspring)

    def next_generation(self):

        self.IndividualInsert(len(self.MSF))
        if self.verbose and self.currentGeneration % 1 == 0:
            if not self.bestFitness == self.msfvector.getFitnesses(self.Tr_x, self.Tr_y):
                print(self.bestFitness)

            print(
                "> Gen #%2d:  Fitness: %.6f //train: %.6f //test: %.6f//dim: %.6f // eval: %.6f " % (
                    self.currentGeneration, self.msfvector.getFitnesses(self.Tr_x, self.Tr_y),
                    self.msfvector.getTrainingMeasure(self.Tr_x, self.Tr_y),
                    self.msfvector.getTestMeasure(self.Te_x, self.Te_y), self.msfvector.getNumberOfDimensions(),
                    self.num_evaluation))

    def updatePopulation(self, offspring):

        [popIndex, createNiches, success, _, fitness] = self.nichingIdentification(offspring)

        if createNiches:
            msf_temp = self.MSF.copy()
            self.MSF.append(offspring)
            self.creat_new_msf_vector(msf_temp, self.MSF, fitness)
        else:
            if success == 1:
                msf_temp = self.MSF.copy()
                self.MSF[popIndex] = offspring
                self.creat_new_msf_vector(msf_temp, self.MSF, fitness)

    def nichingIdentification(self, offspring):
        success = 0
        score = 0
        fitness = 0
        create_niches = False
        pop_index = 0
        if self.fitnessType == "Hing":
            base1 = 0.4
            base2 = 0.3
        else:
            if self.classification:
                base1 = self.base1
                base2 = self.base2
            else:
                if self.dataset_name == "bm1":
                    base1 = 0.01
                    base2 = 0.01
                else:
                    base1 = 0.1
                    base2 = 0.1
        s, fitness_add = self.synergyByAdding(offspring)
        s = s - base1
        sv, fitness_rep = self.synergyByReplacement(offspring)
        sv = np.array(sv) - base2
        if max(sv) <= 0 and s <= 0:
            pop_index = np.argmax(sv)
            create_niches = False
            success = 0
            score = max(sv)
            fitness = max(fitness_rep)
        elif max(sv) > 0:
            if s > max(sv):
                pop_index = len(self.allPopulation)
                create_niches = True
                success = 1
                score = s
                fitness = fitness_add
            else:
                pop_index = np.argmax(sv)
                create_niches = False
                success = 1
                score = max(sv)
                fitness = max(fitness_rep)
        elif s > max(sv):
            if len(self.allPopulation) < self.max_niche_count:
                pop_index = len(self.allPopulation)
                create_niches = True
                success = 1
                score = s
                fitness = fitness_add
            else:
                pop_index = np.argmax(sv)
                create_niches = False
                score = max(sv)
                fitness = max(fitness_rep)
                if max(sv) < 0:
                    success = 0
                else:
                    success = 1
        self.num_evaluation = self.num_evaluation + (self.kfold * len(sv)) + self.kfold

        return pop_index, create_niches, success, score, fitness

    def synergyByReplacement(self, offspring):
        s_rep = []
        fitness_Rep = []
        for i in range(len(self.MSF)):
            msf_temp = self.MSF.copy()
            msf_temp[i] = offspring
            msf_tmp_vector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
                                                     self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name,
                                                     self.classification)
            msf_tmp_vector.create(msf_temp, len(msf_temp))
            s_repi = msf_tmp_vector.getFitnesses(self.Tr_x, self.Tr_y)
            fitness_ = s_repi
            s_repi = ((s_repi - self.bestFitness) / self.bestFitness) * 100
            if self.fitnessType == 'Accuracy':
                pass
            else:
                s_repi = -1 * s_repi

            s_rep.append(s_repi)
            fitness_Rep.append(fitness_)

        return s_rep, fitness_Rep

    def synergyByAdding(self, offspring):
        msf_temp = self.MSF.copy()
        msf_temp.append(offspring)
        msf_tmp_vector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
                                                 self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name,
                                                 self.classification)
        msf_tmp_vector.create(msf_temp, len(msf_temp))
        s_add = msf_tmp_vector.getFitnesses(self.Tr_x, self.Tr_y)
        fitness_ = s_add

        s_add = ((s_add - self.bestFitness) / self.bestFitness) * 100
        if self.fitnessType == 'Accuracy':
            pass
        else:
            s_add = -1 * s_add

        return s_add, fitness_
