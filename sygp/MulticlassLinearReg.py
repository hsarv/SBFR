import numpy as np
import pandas as pd

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
    return e_x / e_x.sum(axis=1, keepdims=True)
class MulticlassLinearReg:

	residuals = None
	classCentroids = None
	coef_ = None

	def __init__(self):
		pass

	def fit(self,X,Y):
		B, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
		self.residuals=residuals
		self.coef_ = B



	def predict(self, X):
		C_test_logit = X @ self.coef_
		logits = C_test_logit.to_numpy()
		probabilities = softmax(logits )
		probability_df = pd.DataFrame(probabilities)
		# Determine the predicted class for each instance
		predicted_classes = probability_df.idxmax(axis=1)
		return predicted_classes

	def predict_proba(self, X):
		C_test_logit = X @ self.coef_
		logits = C_test_logit.to_numpy()
		probabilities = softmax(logits )
		probability_df = pd.DataFrame(probabilities)
		return probability_df