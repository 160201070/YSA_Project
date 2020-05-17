from preprocess import Preprocess
from neural_network import NeuralNetwork

if __name__ == "__main__":
    # Initialize the single neuron neural network
    neural_network = NeuralNetwork()
    preprocess = Preprocess()

    training_inputs = preprocess.get_training_inputs()
    training_outputs = preprocess.get_training_outputs()

    # Train the neural network
    neural_network.train(training_inputs, training_outputs, 10000)

    predicted_outputs = neural_network.think(preprocess.get_training_inputs())
    print("predicted_outputs: ")
    print(predicted_outputs)

    print("expected outputs: ")
    print(preprocess.get_validate_outputs())
