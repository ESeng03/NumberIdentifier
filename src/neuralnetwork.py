import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate):
        #creates an array of weights for the values in layer_sizes ex: (3:4:2) gives [4,3] and [2:4]
        weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:], layer_sizes[:-1])]
        
        #fills the arrays with random values between 1 and 0
        self.weights = [np.random.standard_normal(s)/s[1]**.5 for s in weight_shapes]
        
        #zero array for biases
        self.biases = [np.zeros((s,1)) for s in layer_sizes[1:]]
        
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
        
    