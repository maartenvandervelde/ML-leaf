import numpy as np

class NeuralNetwork():
    def __init__(self, input, output, nHiddenNodes, seed=1):
        np.random.seed(seed)

        if not len(input) == len(output):
            print("Number of training inputs does not match number of training outputs!")
            return

        self.hiddenWeights = 2 * np.random.random((len(input[0]), nHiddenNodes)) - 1
        self.outputWeights = 2 * np.random.random((nHiddenNodes, len(output[0]))) - 1

        self.input = input
        self.output = output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivative(self, x):
        return x * (1 - x);

    def forward(self, x):
        hiddenLayer = self.sigmoid(np.dot(x, self.hiddenWeights))
        outputLayer = self.sigmoid(np.dot(hiddenLayer, self.outputWeights))
        return hiddenLayer, outputLayer
    
    def delta(self, error, output):
        return error * self.sigmoidDerivative(output)

    def drift(self, hidden, output, targetOutput):
        outputError = targetOutput - output
        outputDelta = self.delta(outputError, output)
        outputDrift = np.dot(hidden.T, outputDelta)

        hiddenError = np.dot(outputDelta, self.outputWeights.T)
        hiddenDelta = self.delta(hiddenError, hidden)
        hiddenDrift = np.dot(self.input.T, hiddenDelta)

        return hiddenDrift, outputDrift

    def train(self, epochs, showProgress=False):
        for i in xrange(epochs):
            hiddenLayer, outputLayer = self.forward(self.input);
            hiddenDrift, outputDrift = self.drift(hiddenLayer, outputLayer, self.output)

            self.hiddenWeights += hiddenDrift
            self.outputWeights += outputDrift

            if showProgress:
                progress = (float(i + 1) / float(epochs)) * 100.0
                print("Learned: " + str(int(progress)) + "%")

if __name__ == "__main__":
    input = np.array([[1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 1, 1]])
    output = np.array([[0], [1], [0], [1], [0], [1]])

    nn = NeuralNetwork(input, output, 4)
    nn.train(10000, True)
    result = nn.forward(np.array([1, 0, 1]))[1]
    print(result)