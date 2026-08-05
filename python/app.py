import neuralnet as nn 
import numpy as np


model_path = "../models/pre_trained_mnist.nn"

model = nn.NeuralNetwork(model_path)

print(model)

test_data = np.random.rand(784)

print(model.guess(test_data))
