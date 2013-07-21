class Experiment:
	def __init__(self, training_set, testing_set, Features):
		self.training_set = training_set
		self.testing_set = testing_set
		self.Features = Features
	def results(self):
		results = {}
		return results