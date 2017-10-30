import numpy as np


class FaceRecognizer:
	def __init__(self):
		pass

	def train(self, trainSet, labels):
		imgLen = trainSet.shape[1]
		avg = np.average(trainSet, axis=0)
		trainSet = trainSet.astype(np.float64)
		trainSet -= np.repeat(np.reshape(avg, (1, imgLen)), len(trainSet), 0)
		cov = np.matmul(trainSet, trainSet.transpose())
		eigValues, eigVectors = np.linalg.eig(cov)
		max_arg = eigValues.argsort()[::-1]
		eigValues = eigValues[max_arg[:20]]
		eigVectors = np.matmul(trainSet.transpose(), eigVectors[:, max_arg[:20]])
		self.eigValues = eigValues
		self.eigVectors = eigVectors
		self.coefficients = np.matmul(trainSet, eigVectors)
		self.labels = labels
		self.avg = avg
		self.imLen = imgLen

	def recognize(self, img):
		img = np.reshape(img, (1, self.imLen)) - self.avg
		coef = np.matmul(img, self.eigVectors)
		diff = self.coefficients - np.repeat(coef, len(self.coefficients), 0)
		diff = np.sum(diff ** 2, 1)
		idx = np.argsort(diff)[0]
		return self.labels[idx]

	def getEigVecs(self):
		return self.eigVectors

	def getAvg(self):
		return self.avg

	def getCoefficients(self):
		return self.coefficients
