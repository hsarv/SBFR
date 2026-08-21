import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from .Individual import Individual
from .MostSynergisticFeatures import MostSynergisticFeatures
from .GeneticOperators import getOffspring, GetRep
import time
from random import Random
from sklearn.metrics import pairwise_distances

class ClassifierNotTrainedError(Exception):
	""" You tried to use the classifier before training it. """

def __init__(self, expression, message = ""):
	self.expression = expression
	self.message = message
class SYGP:
	classification=None
	operators = None
	method_name=None
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
	Max_Niche_Size = None
	max_niche_count = None
	terminals = None
	population = None
	allPopulation = None
	pool_population=None
	currentGeneration = 1
	random_pop = None
	standardAproach= None
	OnePop = None
	OnePopSetFitness=None
	bestIndividual: Individual = None
	MSF = None
	msfvector = None
	fitnessOverTime = None

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
		if self.population == None:
			raise ClassifierNotTrainedError("The classifier must be trained using the fit(Tr_X, Tr_Y) method before being used.")

	def __init__(self, allPopulation, population, MSF, operators=[("+", 2), ("-", 2), ("*", 2), ("/", 2)], max_initial_depth=6, population_size=1,
		max_generation=100, max_evaluation=50000, tournament_size=5, elitism_size=1, max_depth=17,
		dim_min=1, dim_max=9999, threads=1, random_state=42, random_pop=False, update_all=False, crossover_type=0, validation=True,kfold=5, verbose=True, model_name="MahalanobisDistanceClassifier", fitnessType="Accuracy", dataset="heart.csv", max_niche_size=300, max_niche_count=30, standardAproach=False, onepop = False, method_name="",OnePopSetFitness=True,n2complexity=0,initial_random=1, classification=True,k_split_trainingset=False):
		self.validation = validation
		self.method_name = method_name
		self.kfold = kfold
		self.classification = classification
		self.random_pop = random_pop
		self.k_split_trainingset=k_split_trainingset
		if sum([0 if op in [("+", 2), ("-", 2), ("*", 2), ("/", 2)] else 0 for op in operators]) > 0:
			print("[Warning] Some of the following operators may not be supported:", operators)
		self.crossover_type = crossover_type
		self.update_all = update_all
		self.operators = operators
		self.max_initial_depth = max_initial_depth
		self.population_size = population_size
		self.threads = max(1, threads)
		self.random_state = random_state
		self.rng = Random(random_state)
		self.max_depth = max_depth
		self.max_generation = max_generation
		self.max_evaluation = max_evaluation
		self.tournament_size = tournament_size
		self.elitism_size = elitism_size
		self.dim_min = max(1, dim_min)
		self.dim_max = max(1, dim_max)
		self.model_name = model_name
		self.fitnessType = fitnessType
		self.verbose = verbose
		self.dataset_name = dataset
		self.Max_Niche_Size = max_niche_size
		self.max_niche_count = max_niche_count
		self.correlationType = "mi"
		self.standardAproach= standardAproach
		self.OnePop=onepop
		self.OnePopSetFitness=OnePopSetFitness
		self.initial_random = initial_random
		self.allPopulation =allPopulation
		self.population = population
		self.MSF = MSF
		self.N2complexity = np.array(n2complexity)

		self.total_rate_cross = 0
		self.total_rate_mute = 0
		self.success_rate_cross = 0
		self.success_rate_mute = 0
		self.validationDecrease = False
		self.msf_temp=None

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
		return "Population Not Trained" if self.msfvector == None else self.msfvector.predict(dataset)

	def convert1DClass(self, y):

		Y = np.zeros((len(y), 1))
		Y = np.argmax(np.max(y, axis=0))

		return Y
	def convert2MultiClass(self,y):
		y = y.reset_index()
		y = y.iloc[:, 1]
		numClass = len(pd.unique(y))
		Y = np.zeros((len(y), numClass))
		for i in range(len(y)):
			for j in range(numClass):
				if j == y[i]:
					Y[i, j] = 1
				else:
					Y[i, j] = 0
		return Y
	def stoppingCriteria(self):

		if self.num_evaluation < 2000:
			perfectTraining=0
			evalLimit=0
			Stock = False
		else:
			Stock = False
			if len(self.fitnessOverTime)>15:
				arr=self.fitnessOverTime[-15:]
				if len(np.unique(arr))==1:
					Stock=True
					print('stock')
			evalLimit=self.num_evaluation >= self.max_evaluation
			if self.msfvector.getFitnesses(self.Tr_x, self.Tr_y) == 1 or self.msfvector.getTrainingMeasure(self.Tr_x,
																										   self.Tr_y) == 1 or self.reach_complexity():
				perfectTraining = 1
				print('perfectTraining')
			else:
				perfectTraining = 0
		if self.validationDecrease:
				self.msfvector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,													 self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name)
				self.msfvector.create(self.msf_temp, len(self.msf_temp))
				print('validationDecrease')
		return perfectTraining or evalLimit or self.validationDecrease or Stock

	def create_pool(self,pool_size):
		for i in range(pool_size):

				ind = Individual(self.operators, self.terminals, self.max_depth, self.model_name, self.fitnessType)
				ind.create(self.rng, n_dims=self.dim_min)
				self.pool_population.append(ind)

	def comp_fitness_indiv(self, ind):
		hyper_x = ind.convert(self.Tr_x)
		mi_y = self.relationwclass(hyper_x, self.Tr_y)
		size = ind.getSize()
		return hyper_x, mi_y[0], size

	def set_fitness_indiv(self, mi_y, size, m_info, i, j):
		self.allPopulation[i][j].setFitness(0.8 * mi_y - (0.1 * size / 100))
		self.allPopulation[i][j].setm_info(m_info)
		self.allPopulation[i][j].setm_iy(mi_y)
	def set_residual(self):
		if len(self.population)==0 and self.initial_random:
			self.residual = self.Tr_y
		else:
			self.residual = self.msfvector.getResidual(self.Tr_x, self.residual)

	def creat_new_msf_vector(self,msf_temp,MSF,fitness=None):
		self.msfvector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
												 self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name)
		self.msfvector.create(MSF, len(MSF))
		if fitness is not None:
			self.bestFitness =fitness
		else:
			self.bestFitness = self.msfvector.getFitnesses(self.Tr_x, self.Tr_y)

		self.set_residual()
		self.msfvector.getTrainingMeasure(self.Tr_x, self.Tr_y)
		self.validationvector.append(self.msfvector.getTestMeasure(self.V_x, self.V_y))
		if (self.validationvector[-1] - self.validationvector[-2]) > 0:
			self.validationDir.append(1)
		elif (self.validationvector[-1] - self.validationvector[-2]) == 0:
			self.validationDir.append(0)
		else:
			self.validationDir.append(-1)
		#------
		neg = 0
		length_of_search = min(len(self.validationDir), 4)
		for i in range(length_of_search):
			if self.validationDir[-1 - i] == 1:
				break
			elif self.validationDir[-1 - i] == 0:
				pass
			else:
				neg = neg + 1
		if neg > 2:
			self.validationDecrease = True
			self.MSF = self.msf_temp.copy()
			self.msfvector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
													 self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name)
			self.msfvector.create(MSF, len(MSF))
		else:
			self.msf_temp = self.MSF.copy()


	def fit(self,Tr_x, Tr_y, Te_x = None, Te_y = None, V_x=None, V_y=None):
		if self.verbose:
			print("  > Parameters")
			print("    > SYGP Type:       "+str(self.method_name))
			print("    > Random State:       "+str(self.random_state))
			print("    > Random pop:       " + str(self.random_pop))
			print("    > Tournament Size:    "+str(self.tournament_size))
			print("    > Validation:         "+str(self.validation))
			print("    > CrossoverType:         " + str(self.crossover_type))
			print("    > Wrapped Model:      "+self.model_name)
			print("    > dataset_name:      " + self.dataset_name)
			print("    > K fold:            " + str(self.kfold))
			print("    > population size:            "+str(self.population_size))
			print()

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
		self.create_pool(100)
		self.set_residual()

		if self.initial_random == 1:
			ind = Individual(self.operators, self.terminals, self.max_depth, self.model_name, self.fitnessType)
			ind.create(self.rng, n_dims=self.dim_min)
			self.population.append(ind)
			self.allPopulation.append(self.population)
			self.MSF.append(self.population[0])
			self.creat_new_msf_vector(self.MSF,self.MSF)

			if not (self.OnePop):
				self.allPopulation[-1][-1].isRep = True
			if not self.random_pop:
				self.allPopulation[-1][-1].set_Age(0)
				if self.OnePopSetFitness:
					hyper_x, mi_y, size = self.comp_fitness_indiv(self, ind)
					self.set_fitness_indiv( mi_y, size, 0, -1, -1)

			self.IndividualInsert(self.standardAproach, self.OnePop, self.population_size,0, self.initial_random)
		else:
			self.creat_new_msf_vector(self.MSF,self.MSF)


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
				t1 = time.time()
				self.nextGeneration(self.currentGeneration)
				t2 = time.time()
				duration = t2-t1
			else:
				self.num_evaluation = self.max_evaluation
				duration = 0
			if overtime:
				self.evalOverTime.append(self.num_evaluation)
				self.fitnessOverTime.append(self.msfvector.getFitnesses(self.Tr_x, self.Tr_y))
				self.trainingAccuracyOverTime.append(self.msfvector.getTrainingMeasure(self.Tr_x,self.Tr_y))
				self.testAccuracyOverTime.append(self.msfvector.getTestMeasure(self.Te_x, self.Te_y))

				self.dimOverTime.append(self.msfvector.getNumberOfDimensions())
				self.currentGeneration += 1
		if overtime:
			self.evalOverTime.append(self.num_evaluation)
			self.fitnessOverTime.append(self.msfvector.getFitnesses(self.Tr_x, self.Tr_y))
			self.trainingAccuracyOverTime.append(self.msfvector.getTrainingMeasure(self.Tr_x,self.Tr_y))
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
	def findoffspring(self, offspring):
		hyper_x1, mi_y1, size1 = self.comp_fitness_indiv(self, offspring)
		idx1, minfo1 = self.mutualInfoBFeatures(offspring)

		if not len(offspring) == 1:
			hyper_x2, mi_y2, size2 = self.comp_fitness_indiv(self, offspring)
			idx2, minfo2 = self.mutualInfoBFeatures(offspring)

			if mi_y1 < 0.001:
				off = offspring[1]
				minfo_end = minfo2
				idx = idx2
				mi_yy = mi_y2
			else:
				if mi_y1 > mi_y2:
					if minfo1 - minfo2 < 0.2:
						off = offspring[0]
						minfo_end = minfo1
						idx = idx1
						mi_yy = mi_y1
					else:
						if mi_y2 > 0.001:
							off = offspring[1]
							mi_yy = mi_y2
							minfo_end = minfo2
							idx = idx2
						else:
							off = offspring[0]
							mi_yy = mi_y1
							minfo_end = minfo1
							idx = idx1
				else:
					if minfo2 - minfo1 < 0.2:
						off = offspring[1]
						mi_yy = mi_y2
						minfo_end = minfo2
						idx = idx2
					else:
						off = offspring[0]
						mi_yy = mi_y1
						minfo_end = minfo1
						idx = idx1
		else:
			off = offspring[0]
			minfo_end = minfo1
			idx = idx1
			mi_yy = mi_y1
		return off,minfo_end,idx,mi_yy

	# if self.crossover_type == 1 or self.crossover_type == 2:
	# 	self.increaseAgeMulti()
	# 	rep_i = self.MSF[i]
	# 	idd, minf = self.minMutualInfoBFeatures(rep_i)
	# else:

	def IndividualInsert(self,standardAproach, OnePop, Popsize,numGeneration,randomPop):
		if not standardAproach:
			for i in range(Popsize):
				if randomPop:
					offspring = Individual(self.operators, self.terminals, self.max_depth, self.model_name, self.fitnessType)
					offspring.create(self.rng, n_dims=self.dim_min)
					if self.OnePopSetFitness:
						hyper_x, mi_y, size = self.comp_fitness_indiv(self, offspring)
						idx, minfo = self.mutualInfoBFeatures(offspring)
						self.updatePopulation(offspring, minfo, mi_y, idx, size,0)
					else:
						self.updatePopulation(offspring, 0, 0, 0, 0, 0)
				else:
					idd=0
					offspring, id1, id2, iscross, p1, p2 = getOffspring(self.pool_population,self.rng, self.allPopulation, i, numGeneration, self.tournament_size, self.Tr_x, self.Tr_y, self.crossover_type, idd, self.operators, self.terminals, self.max_depth, self.model_name, self.fitnessType,self.residual)
					if self.OnePopSetFitness:
						off, minfo_end, idx, mi_yy = self.findoffspring(offspring)
						size = off.getSize()
						self.updatePopulation(off, minfo_end, mi_yy, idx, size, iscross)
					else:
						self.updatePopulation(offspring[0], 0, 0, 0, 0, iscross)
					self.deleteIndividualMulti(100)

		else:
			for i in range(Popsize):
				if randomPop:
					offspring = Individual(self.operators, self.terminals, self.max_depth, self.model_name,
									   self.fitnessType)
					offspring.create(self.rng, n_dims=self.dim_min)

					if OnePop:
						if not self.random_pop:
							if self.OnePopSetFitness:
								hyper_x, mi_y, size = self.comp_fitness_indiv(self, offspring)
								idx, minfo = self.mutualInfoBFeatures(offspring)

								evaluation=True
								if evaluation:
									self.standardOnePopUpdatePopulation(offspring, minfo, mi_y, idx,size,0,0)
								else:
									self.allPopulation[-1].append(offspring)
									self.allPopulation[-1][-1].setFitness(0.8*mi_y[0]-(0.1*size/100))
									self.allPopulation[-1][-1].setm_info(0)                        
									self.allPopulation[-1][-1].setm_iy(mi_y[0])
									self.allPopulation[-1][-1].set_Age(0)
									self.allPopulation[-1].sort(reverse=True)
							else:
								self.standardOnePopUpdatePopulation(offspring, 0, 0, 0, 0, 0, 0)
						else:
							self.standardOnePopUpdatePopulation(offspring, 0, 0, 0,0 ,0,0)
					else:
						hyper_x, mi_y, size = self.comp_fitness_indiv(self, offspring)
						idx, minfo = self.mutualInfoBFeatures(offspring)
						self.standardUpdatePopulation(offspring, minfo, mi_y, idx,size,0)
				else:
					if not self.OnePop:
						rep_i = self.MSF[i]
						idd, minf = self.minMutualInfoBFeatures(rep_i)
						offspring, id1, id2, isCross, p1, p2 = getOffspring(self.rng, self.allPopulation, i, numGeneration, self.tournament_size, self.Tr_x, self.Tr_y, self.crossover_type, idd, self.operators, self.terminals, self.max_depth, self.model_name, self.fitnessType,self.residual )
						hyper_x1, mi_y1, size1 = self.comp_fitness_indiv(self, offspring[0])
						idx1, minfo1 = self.mutualInfoBFeatures(offspring)

						if not len(offspring) == 1:
							iscross = 1
							hyper_x2, mi_y2, size2 = self.comp_fitness_indiv(self, offspring[1])
							idx2, minfo2 = self.mutualInfoBFeatures(offspring[1])

							if mi_y1-mi_y2 > 0.09:
								if size1-size2 < 5:
									off = offspring[0]
									mi_yy = mi_y1
									minfo_end = minfo1
									idx = idx1
									size = size1
								else:
									off = offspring[1]
									mi_yy = mi_y2
									minfo_end = minfo2
									idx = idx2
									size = size2
							else:
								if size1-size2 < 5:
									off = offspring[0]
									mi_yy = mi_y1
									minfo_end = minfo1
									idx = idx1
									size = size1
								else:
									off = offspring[1]
									mi_yy = mi_y2
									minfo_end = minfo2
									idx = idx2
									size = size2
						else:
							off = offspring[0]
							minfo_end = minfo1
							idx = idx1
							mi_yy = mi_y1
							size = size1
							iscross = 0

						self.standardUpdatePopulation(off, minfo_end, mi_yy, idx, size, iscross)
					else:
						self.increaseAge()
						offspring, id1, id2, isCross, p1, p2 = getOffspring(self.rng, self.allPopulation, 0, numGeneration, self.tournament_size, self.Tr_x, self.Tr_y, self.crossover_type, 0, self.operators, self.terminals, self.max_depth, self.model_name, self.fitnessType, self.residual)
						off=offspring[0]
						if self.OnePopSetFitness:
							hyper_x1, mi_y1, size1 = self.comp_fitness_indiv(self, offspring[0])
							idx1, minfo1 = self.mutualInfoBFeatures(offspring[0])
							size = size1
							off = offspring[0]
							mi_yy = mi_y1
							minfo_end = minfo1
							idx = idx1
							if not len(offspring) == 1:
								hyper_x2, mi_y2, size2 = self.comp_fitness_indiv(self, offspring[1])
								idx2, minfo2 = self.mutualInfoBFeatures(offspring[1])
								f1 = 0.8*mi_y1[0]-0.1*minfo1-(0.1*size1/100)
								f2 = 0.8*mi_y2[0]-0.1*minfo2-(0.1*size2/100)
								if f1 < f2:
									off = offspring[1]
									minfo_end = minfo2
									idx = idx2
									mi_yy = mi_y2
									size = size2
							self.standardOnePopUpdatePopulation(off, minfo_end ,mi_yy, idx, size, 0, 0)
						else:
							self.standardOnePopUpdatePopulation(off, 0,0, 0,0, 0, 0)

						self.deleteIndividual(100)

	def nextGeneration(self, numGeneration):

		begin = time.time()
		self.IndividualInsert(self.standardAproach, self.OnePop, len(self.MSF), numGeneration,self.random_pop)
		end = time.time()
		if self.verbose and self.currentGeneration % 1 == 0:
			#print(self.msfvector.getTestMeasure(self.V_x, self.V_y))
			print(
			"> Gen #%2d:  Fitness: %.6f //train: %.6f //test: %.6f//val: %.6f // dim: %.6f // eval: %.6f // Time: %.4f // resi: %.6f" % (
				self.currentGeneration, self.msfvector.getFitnesses(self.Tr_x,self.Tr_y), self.msfvector.getTrainingMeasure(self.Tr_x,self.Tr_y),
				self.msfvector.getTestMeasure(self.Te_x, self.Te_y),self.msfvector.getTestMeasure(self.V_x, self.V_y), self.msfvector.getNumberOfDimensions(),
				self.num_evaluation, end - begin, np.sum(self.residual * self.residual)))
			#print(self.msfvector)
	def increaseAge(self):
		for i in range(len(self.allPopulation[-1])):
			newAge = self.allPopulation[-1][i].getAge() + 1
			self.allPopulation[-1][i].set_Age(newAge)

	def increaseAgeMulti(self):
		for j in range(len(self.allPopulation)):
			for i in range(len(self.allPopulation[j])):
				newAge = self.allPopulation[j][i].getAge() + 1
				self.allPopulation[j][i].set_Age(newAge)

	def deleteIndividual(self, maxAge):
		if len(self.allPopulation[-1]) > maxAge:
			notdeleteIndiv=[]
			delete=False
			for i in range(len(self.allPopulation[-1])):
				newAge = self.allPopulation[-1][i].getAge() + 1
				notdeleteIndiv.append(i)
				if newAge > maxAge:
					delete = True
					notdeleteIndiv.remove(i)
			if delete:
				pop = self.allPopulation[0].copy()
				self.allPopulation[0] = []
				for k in range(len(notdeleteIndiv)):
					self.allPopulation[0].append(pop[notdeleteIndiv[k]])

	def deleteIndividualMulti(self, maxAge):
		for j in range(len(self.allPopulation)):
			if len(self.allPopulation[j]) > maxAge:
				notdeleteIndiv=[]
				delete = False
				for i in range(len(self.allPopulation[j])):
					newAge = self.allPopulation[j][i].getAge() + 1
					notdeleteIndiv.append(i)
					if newAge > maxAge:
						if not self.allPopulation[j][i].isRep:
							delete = True
							notdeleteIndiv.remove(i)
				if delete:
					pop = self.allPopulation[j].copy()
					self.allPopulation[j] = []
					for k in range(len(notdeleteIndiv)):
						self.allPopulation[j].append(pop[notdeleteIndiv[k]])

	def update_total_genetic_operator_rate(self,iscross):
		if iscross:
			self.total_rate_cross = self.total_rate_cross + 1
		else:
			self.total_rate_mute = self.total_rate_mute + 1

	def update_genetic_operator_rate(self, iscross):
		if iscross:
			self.success_rate_cross = self.success_rate_cross + 1
		else:
			self.success_rate_mute = self.success_rate_mute + 1

	def add_pool_population(self,offspring):
		self.pool_population.append(offspring)
		if len(self.pool_population) > 300:
			del self.pool_population[0]

	def updatePopulation(self, offspring,mi, mi_y,pi,size,iscross):
		[popIndex, createNiches, success, score, fitness] = self.nichingIdentification(offspring)
		self.update_total_genetic_operator_rate(iscross)
		self.add_pool_population(offspring)

		if createNiches:
			self.update_genetic_operator_rate(iscross)
			new_population = []
			new_population.append(offspring)
			self.msf_temp = self.MSF.copy()
			self.allPopulation.append(new_population)
			self.MSF.append(offspring)
			self.creat_new_msf_vector(self.msf_temp, self.MSF, fitness)
			if not self.random_pop:
				self.allPopulation[-1][-1].isRep = True
				if self.OnePopSetFitness:
					self.set_fitness_indiv(self, mi_y, size, mi, popIndex, -1)
					self.allPopulation[popIndex][-1].set_Age(0)
					for ii in range(len(self.allPopulation) - 1):
						notdeleteIndiv = []
						can, id = GetRep(self.allPopulation[ii])
						notdeleteIndiv.append(id)
						for j in range(len(self.allPopulation[ii])):
							if not id == j:
								idx, minfo = self.mutualInfoBFeatures(self.allPopulation[ii][j])
								if not idx == ii:
									self.allPopulation[idx].append(self.allPopulation[ii][j])
									if self.OnePopSetFitness:
										self.allPopulation[idx][-1].setm_info(minfo)
										m_y = self.allPopulation[ii][j].mi_y
										size = self.allPopulation[ii][j].size
										self.allPopulation[idx][-1].setFitness(
											0.8 * m_y - 0.1 * minfo - (0.1 * size / 100))
								else:
									notdeleteIndiv.append(j)
						pop = self.allPopulation[ii].copy()
						self.allPopulation[ii] = []
						for k in range(len(notdeleteIndiv)):
							self.allPopulation[ii].append(pop[notdeleteIndiv[k]])
					if self.tournament_size > 1:
						for ii in range(len(self.allPopulation)):
							self.allPopulation[ii].sort(reverse=True)
		else:
			if not success == 1:
				if self.OnePopSetFitness:
					if mi_y > 0.001:
						if not popIndex == pi:
							popIndex = pi
						self.allPopulation[popIndex].append(offspring)
						self.allPopulation[popIndex][-1].set_Age(1)
						if self.OnePopSetFitness:
							self.set_fitness_indiv(self, mi_y, size, mi, popIndex, -1)
							self.allPopulation[popIndex].sort(reverse=True)
					else:
						self.pool_population.remove(-1)

			if success == 1:
				self.update_genetic_operator_rate(iscross)
				self.allPopulation[popIndex].append(offspring)
				self.allPopulation[popIndex][-1].set_Age(0)
				self.msf_temp = self.MSF.copy()
				self.MSF[popIndex] = offspring
				self.creat_new_msf_vector(self.msf_temp, self.MSF, fitness)
				can, id = GetRep(self.allPopulation[popIndex])
				self.allPopulation[popIndex][id].isRep = False
				self.allPopulation[popIndex][-1].isRep = True
				if self.OnePopSetFitness:
					self.set_fitness_indiv(self, mi_y, size, 0, popIndex, -1)
					if len(self.allPopulation) > 1:
						notdltindiv = []
						notdltindiv.append(len(self.allPopulation[popIndex]) - 1)
						for i in range(len(self.allPopulation[popIndex]) - 1):
							idx, minfo = self.mutualInfoBFeatures(self.allPopulation[popIndex][i])
							if not idx == popIndex:
								self.allPopulation[idx].append(self.allPopulation[popIndex][i])
								if self.OnePopSetFitness:
									self.allPopulation[idx][-1].setm_info(minfo)
									m_y = self.allPopulation[popIndex][i].mi_y
									size = self.allPopulation[popIndex][i].size
									self.allPopulation[idx][-1].setFitness(0.8 * m_y - 0.1 * minfo - (0.1 * size / 100))
							else:
								notdltindiv.append(i)
								self.allPopulation[idx][i].setm_info(minfo)
						if not len(notdltindiv) == len(self.allPopulation[popIndex]):
							pop = self.allPopulation[popIndex].copy()
							self.allPopulation[popIndex] = []
							for ii in range(len(notdltindiv)):
								self.allPopulation[popIndex].append(pop[notdltindiv[ii]])
					if self.tournament_size > 1:
						for ii in range(len(self.allPopulation)):
							self.allPopulation[ii].sort(reverse=True)

	def standardUpdatePopulation(self, offspring,mi, mi_y,pi,size,iscross):
		[popIndex, createNiches, success, score, fitness] = self.nichingIdentification(offspring)
		if createNiches:
			new_population = []
			new_population.append(offspring)
			self.allPopulation.append(new_population)
			self.MSF.append(offspring)
			self.msfvector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
													 self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name)
			self.msfvector.create(self.MSF, len(self.MSF))

			self.bestFitness = fitness
			self.msfvector.getTrainingMeasure(self.Tr_x, self.Tr_y)
			self.validationvector.append(self.msfvector.getTestMeasure(self.V_x, self.V_y))
			if (self.validationvector[-1]-self.validationvector[-2])>0 or (self.validationvector[-1]-self.validationvector[-2])==0 :
				self.validationDir.append(1)
			else:
				self.validationDir.append(-1)

			self.residual = self.msfvector.getResidual((self.Tr_x, self.residual))
			self.allPopulation[-1][-1].isRep = True
			self.allPopulation[popIndex][-1].setFitness(0.8*mi_y-(0.1*size/100))
			self.allPopulation[popIndex][-1].setm_info(0)
			self.allPopulation[popIndex][-1].setm_iy(mi_y)
		else:
			if len(self.allPopulation[popIndex]) + 1 > self.Max_Niche_Size and (not self.allPopulation[popIndex][
				-1].isRep):
				del self.allPopulation[popIndex][-1]
			if not success == 1:
				self.allPopulation[popIndex].append(offspring)
				self.allPopulation[popIndex][-1].setFitness(0.8*mi_y-0.1*mi-(0.1*size/100))
				self.allPopulation[popIndex][-1].setm_info(mi)
				self.allPopulation[popIndex][-1].setm_iy(mi_y)
			if success == 1:
				mi = 0
				self.allPopulation[popIndex].append(offspring)
				self.allPopulation[popIndex][-1].setFitness(0.8*mi_y-0.1*mi-(0.1*size/100))
				self.allPopulation[popIndex][-1].setm_info(mi)
				self.allPopulation[popIndex][-1].setm_iy(mi_y)
				self.MSF[popIndex] = offspring
				self.msfvector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
														 self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name)
				self.msfvector.create(self.MSF, len(self.MSF))
				can, id = GetRep(self.allPopulation[popIndex])
				self.allPopulation[popIndex][id].isRep = False
				self.allPopulation[popIndex][-1].isRep = True

				self.bestFitness = fitness
				self.validationvector.append(self.msfvector.getTestMeasure(self.V_x, self.V_y))
				if (self.validationvector[-1] - self.validationvector[-2]) > 0 or (
						self.validationvector[-1] - self.validationvector[-2]) == 0:
					self.validationDir.append(1)
				else:
					self.validationDir.append(-1)
				self.residual = self.msfvector.getResidual(self.Tr_x, self.residual)

		self.allPopulation[popIndex].sort(reverse=True)

	def standardOnePopUpdatePopulation(self, offspring, mi, mi_y, pi, size, iscross,id1):
		[popIndex, createNiches, success, score, fitness] = self.nichingIdentification(offspring)
		if createNiches:
			self.MSF.append(offspring)
			self.msfvector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
													 self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name)
			self.msfvector.create(self.MSF, len(self.MSF))
			self.bestFitness = fitness
			self.validationvector.append(self.msfvector.getTestMeasure(self.V_x, self.V_y))
			if (self.validationvector[-1]-self.validationvector[-2])>0 or (self.validationvector[-1]-self.validationvector[-2])==0 :
				self.validationDir.append(1)
			else:
				self.validationDir.append(-1)
			self.residual = self.msfvector.getResidual(self.Tr_x, self.residual)
			if not self.random_pop:
				self.allPopulation[-1].append(offspring)
				if self.OnePopSetFitness:
					self.allPopulation[-1][-1].setFitness(0.8*mi_y-(0.1*size/100))
					self.allPopulation[-1][-1].setm_info(0)
					self.allPopulation[-1][-1].setm_iy(mi_y)
				self.allPopulation[-1][-1].set_Age(0)
		else:
			if not self.random_pop:
				if not success == 1:
					self.allPopulation[-1].append(offspring)
					if self.OnePopSetFitness:
						self.allPopulation[-1][-1].setFitness(0.8 * mi_y -0.1*mi- (0.1 * size / 100))
						self.allPopulation[-1][-1].setm_info(mi)
						self.allPopulation[-1][-1].setm_iy(mi_y)
					self.allPopulation[-1][-1].set_Age(1)
			if success == 1:
				if not self.random_pop:
					self.allPopulation[-1].append(offspring)
					if self.OnePopSetFitness:
						self.allPopulation[-1][-1].setFitness(0.8 * mi_y-0.1*mi-(0.1 * size / 100))
						self.allPopulation[-1][-1].setm_info(mi)
						self.allPopulation[-1][-1].setm_iy(mi_y)
					self.allPopulation[-1][-1].set_Age(0)
				self.MSF[popIndex] = offspring
				self.msfvector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
														 self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name)
				self.msfvector.create(self.MSF, len(self.MSF))
				self.bestFitness = fitness
				self.validationvector.append(self.msfvector.getTestMeasure(self.V_x, self.V_y))
				if (self.validationvector[-1] - self.validationvector[-2]) > 0 or (
						self.validationvector[-1] - self.validationvector[-2]) == 0:
					self.validationDir.append(1)
				else:
					self.validationDir.append(-1)
				self.residual = self.msfvector.getResidual(self.Tr_x, self.residual)
		if self.OnePopSetFitness:
			self.allPopulation[-1].sort(reverse=True)

	def relationtfeatures(self, r, hyper_x):
		if self.correlationType == "corr":
			mi = r.corrwith(hyper_x)
			mi_y = mi
		else:
			mi_y = mutual_info_regression(r, hyper_x, discrete_features=False)
		return mi_y

	def relationwclass(self, hyper_x, hyper_Y):
		if self.correlationType == "corr":
			df1 = hyper_Y.to_frame()
			df = pd.concat([df1, hyper_x], axis=1, join='outer')
			dict={'0':'Target'}
			df.rename(columns=dict, inplace=True)
			mi = hyper_x.corrwith()
			mi_yy = mi["Y"]
			mi_y = []
			if np.isnan(mi_yy):
				mi_y.append(0)
			else:
				mi_y = mi_yy
			# mi_y, p_value = pearsonr(hyper_x, hyper_Y.to_frame())
		else:
			mi_y = mutual_info_classif(hyper_x, hyper_Y)
		return mi_y


	def mutualInfoBFeatures(self, offspring):
		r = self.msfvector.convert(self.Tr_x)
		hyper_x = offspring.convert(self.Tr_x)
		minfo = pd.DataFrame()
		top_k = r.shape[1]
		minfo =self.relationtfeatures(r, hyper_x)
		top_k_idx = minfo.argsort()[-top_k:][::-1]
		return top_k_idx[0], minfo[top_k_idx[0]]


	def minMutualInfoBFeatures(self, offspring):
		r = self.msfvector.convert(self.Tr_x)
		hyper_x = offspring.convert(self.Tr_x)
		top_k = r.shape[1]
		minfo = self.relationtfeatures(r, hyper_x)
		top_k_idx = minfo.argsort()[-top_k:][::-1]
		return top_k_idx[-1], minfo[top_k_idx[-1]]

	def nichingIdentification(self, offspring):
		pop_index=0
		base = 0.1
		s, fitness_add = self.synergyByAdding(offspring)
		s = s - base
		sv, fitness_rep = self.synergyByReplacement(offspring)
		sv = np.array(sv) - base
		if max(sv) <= 0 and s <= 0:
			pop_index = np.argmax(sv)
			create_niches = False
			success = 0
			score = max(sv)
			fitness = max(fitness_rep)
		elif max(sv) > 0:
			if (s > max(sv)):
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
		if pd.isnull(create_niches):
			print()
		return pop_index, create_niches, success, score, fitness

	def synergyByReplacement(self, offspring):
		s_rep = []
		fitness_Rep = []
		for i in range(len(self.MSF)):
			msf_temp = self.MSF.copy()
			msf_temp[i] = offspring
			msf_tmp_vector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
													 self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name)
			msf_tmp_vector.create(msf_temp, len(msf_temp))
			s_repi = msf_tmp_vector.getFitnesses(self.Tr_x, self.Tr_y)
			fitness = s_repi
			s_repi = ((s_repi - self.bestFitness) / (self.bestFitness)) * 100
			if self.fitnessType=='Accuracy':
				pass
			else:
				s_repi = -1 * (s_repi )

			s_rep.append(s_repi)
			fitness_Rep.append(fitness)

		return s_rep, fitness_Rep

	def selfSynergy(self, offspring):
		msf_temp = []
		msf_temp.append(offspring)
		msf_tmp_vector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
												 self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name)
		msf_tmp_vector.create(msf_temp, len(msf_temp))
		s_self = msf_tmp_vector.getFitnesses(self.Tr_x, self.Tr_y)
		s_self = ((s_self - self.bestFitness) / self.bestFitness) * 100
		return s_self

	def synergyByAdding(self, offspring):
		msf_temp = self.MSF.copy()
		msf_temp.append(offspring)
		msf_tmp_vector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
												 self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name)
		msf_tmp_vector.create(msf_temp, len(msf_temp))
		s_add = msf_tmp_vector.getFitnesses(self.Tr_x, self.Tr_y)
		fitness = s_add

		s_add = ((s_add - self.bestFitness) / (self.bestFitness)) * 100
		if self.fitnessType=='Accuracy' :
			pass
		else:
			s_add = -1*(s_add)

		return s_add, fitness

	def updateFitness(self, allPopulation, success, createNiches, popIndex):
		if createNiches or success == 1:
			MSF_new = []
			for i in range(len(allPopulation)):
				for j in range(len(allPopulation[i])):
					msf_temp= self.MSF.c
					msf_temp[i] = allPopulation[i][j]
					msf_tmp_vector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation,
															 self.kfold, self.Tr_x, self.Tr_y, self.fitnessType,
															 self.dataset_name)
					msf_tmp_vector.create(msf_temp, len(msf_temp))
					fitness = msf_tmp_vector.getFitnesses(self.Tr_x, self.Tr_y)
					if fitness > self.bestFitness:
						# updateState=True
						self.MSF[i] = self.allPopulation[i][j]
						can, id = GetRep(self.allPopulation[i])
						self.allPopulation[i][id].isRep = False
						self.allPopulation[i][j].isRep = True
						self.bestFitness=fitness
						self.residual = self.msfvector.getResidual(self.Tr_x, self.residual)
					self.allPopulation[i][j].setFitness(fitness)
				self.allPopulation[i].sort(reverse=True)
				MSF_new.append(self.allPopulation[i][0])
			self.MSF = MSF_new.copy()
			self.msfvector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation, self.kfold,
													 self.Tr_x, self.Tr_y, self.fitnessType, self.dataset_name)
			self.msfvector.create(self.MSF, len(self.MSF))

			for k in range(len(allPopulation)):
				can, id = GetRep(self.allPopulation[k])
				if not id == 0:
					self.allPopulation[k][id].isRep = False
					self.allPopulation[k][0].isRep = True

		else:
			msf_temp = self.MSF.copy()
			msf_temp[popIndex] = allPopulation[popIndex][-1]
			msf_tmp_vector = MostSynergisticFeatures(self.max_depth, self.model_name, self.validation,
													 self.kfold, self.Tr_x, self.Tr_y, self.fitnessType,
													 self.dataset_name)
			msf_tmp_vector.create(msf_temp, len(msf_temp))
			fitness = msf_tmp_vector.getFitnesses(self.Tr_x, self.Tr_y)
			self.allPopulation[popIndex][-1].setFitness(fitness)



	def compute_2N_complexity(self, X, y):

		for i in range(len(y)):
			if y[i] > 0:
				y[i] = 1
			else:
				y[i] = -1

		# Calculate pairwise distances between all points
		distances = pairwise_distances(X, metric='euclidean')


		# Initialize lists to store nearest neighbor distances
		nearest_within_class = []
		nearest_between_class = []

		# Loop through each point to find the nearest neighbors
		for i in range(len(X)):
			# Mask for the current class
			same_class_mask = (y == y[i])
			other_class_mask = ~same_class_mask
#			# Distances within the same class, excluding the point itself
			within_class_distances = distances[i, same_class_mask]
			within_class_distances = within_class_distances[within_class_distances != 0]

			# Distances to other classes
			between_class_distances = distances[i, other_class_mask]

			# Nearest neighbor distances
			if len(within_class_distances) > 0:
				nearest_within_class.append(np.min(within_class_distances))
			if len(between_class_distances) > 0:
				nearest_between_class.append(np.min(between_class_distances))

		# Calculate the average distances
		avg_within_class = np.mean(nearest_within_class)
		avg_between_class = np.mean(nearest_between_class)

		# Calculate and return the 2N complexity measure
		complexity_2N = avg_within_class / (avg_between_class+0.001)
		return complexity_2N


	def reach_complexity(self):
		# if not(self.N2complexity[-1]==0):
		# 	y_pred = self.predict(self.Tr_x)
		# 	N2_complexity = self.compute_2N_complexity(self.Tr_x, y_pred)
		# 	# if len(self.N2complexity) > 1:
		# 	# 	thr = (self.N2complexity[-1]-self.N2complexity[-2])
		# 	# else:
		# 	# 	thr = 0.03
		# 	thr=0.15
		# 	condition =N2_complexity - self.N2complexity[-1] > thr
		# 	if condition==True:
		# 		print('didi conditiono')
		# else:
		# 	condition=False
		if not (self.N2complexity[-1] == 0):
			if self.num_evaluation >5000:
				condition=True
			else:
				condition = False
		else:
			condition = False
		return condition