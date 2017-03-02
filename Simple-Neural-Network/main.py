import prepare_data as pd
import Neural_Network as nn

def main():

    X_train, Y_train, X_test, Y_test = pd.prepare_data()

    pd.show_data(X_train, Y_train)

    network = nn.Neural_Network()

    number_of_iteration = 1000

    # Learn neural network
    for i in range(number_of_iteration):
        network.learn(X_train, Y_train, 0.1)

    predict = network.forward(X_test)

    predict = nn.Neural_Network.make_binary_output(predict)

    accuracy = nn.Neural_Network.compute_accuracy(predict, Y_test)

    print("Accuracy :", end=" ")
    print(accuracy)

main()