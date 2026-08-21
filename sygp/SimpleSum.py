import numpy as np


class SimpleSum:


	def __init__(self):
		pass

	def fit(self,X,Y):
		pass


	def predict(self, X):		
		predictions1 = []
		X = [ list(sample) for sample in X.iloc ]

		for sample in X:
			pick=np.sum(sample)
			predictions1.append(pick)

		predictions=np.array(predictions1)
		return predictions

	def predict_probability(self, X):
		predictions = []
		X = [ list(sample) for sample in X.iloc ]

		for sample in X:
			pick=np.sum(sample)
			predictions.append(pick)
		return predictions