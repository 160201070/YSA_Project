from preprocess import Preprocess
from neural_network import NeuralNetwork
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Initialize the single neuron neural network
    preprocess = Preprocess()
    neural_network = NeuralNetwork()
    
    test_input = preprocess.get_training_inputs().to_numpy().reshape(60899,11)
    test_output = preprocess.get_training_outputs().to_numpy().reshape(60899,1)
    
# =============================================================================
#     df = df[~df['Movie_Id'].isin(drop_movie_list)]
#     self.dataAll = df_p.iloc[:,5]
#     drop = test_input.iloc[:,5]
# =============================================================================
        
# =============================================================================
#     training_inputs2 = test_input.to_numpy().reshape(60899,11)
#     training_outputs2 = test_output.to_numpy().reshape(60899,1)
# =============================================================================
    
    test_input = test_input.astype(int)
    test_output= test_output.astype(int)
    
    neural_network.train(test_input, test_output, 10000)
    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)
    
    
    predicted_outputs = neural_network.think(preprocess.get_training_inputs())
    print("predicted_outputs: ")
    print(predicted_outputs[:])

      
# =============================================================================
#     training_inputs = preprocess.get_training_inputs()
#     training_outputs = preprocess.get_training_outputs()
# 
#     # Train the neural network
#     neural_network.train(training_inputs, training_outputs, 10000)
# 
#     predicted_outputs = neural_network.think(preprocess.get_training_inputs())
#     print("predicted_outputs: ")
#     print(predicted_outputs)
# 
#     print("expected outputs: ")
#     print(preprocess.get_validate_outputs())
# =============================================================================
