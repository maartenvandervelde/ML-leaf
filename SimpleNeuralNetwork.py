import numpy as np
import sys
import glob
import csv
import os

#The class that constructs a neural network
class NeuralNetwork():
    def __init__(self, nHiddenNodes=0, hiddenWeights=None, outputWeights=None, seed=1):
        #Seed the random function (good practice for back checking)
        np.random.seed(seed)
        
        #See if we use a precalculated set of weights, if not initialize random weights
        if (hiddenWeights is None):
            self.hiddenWeights = 2 * np.random.random((len(input[0]), nHiddenNodes)) - 1
        else:
            self.hiddenWeights = hiddenWeights

        if (outputWeights is None):
            self.outputWeights = 2 * np.random.random((nHiddenNodes, len(output[0]))) - 1
        else:
            self.outputWeights = outputWeights

    #The sigmoid function used for forward propagation to calculate the different layers within the network
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #The derivative of the sigmoid used to calculate the delta
    def sigmoidDerivative(self, x):
        return x * (1 - x);

    #The forward propagation part, calculates the hidden and ouput layers
    def forward(self, x):
        hiddenLayer = self.sigmoid(np.dot(x, self.hiddenWeights))
        outputLayer = self.sigmoid(np.dot(hiddenLayer, self.outputWeights))
        return hiddenLayer, outputLayer
    
    #Calculate the delta error usefull for the backpropagation
    def delta(self, error, output):
        return error * self.sigmoidDerivative(output)

    #Calculate the amount that the weights need to change (here denoted as "drift")
    def drift(self, input, hidden, output, targetOutput):
        outputError = targetOutput - output
        outputDelta = self.delta(outputError, output)
        outputDrift = np.dot(hidden.T, outputDelta)

        hiddenError = np.dot(outputDelta, self.outputWeights.T)
        hiddenDelta = self.delta(hiddenError, hidden)
        hiddenDrift = np.dot(input.T, hiddenDelta)

        return hiddenDrift, outputDrift

    #Combines the above functions to calculate the final weights
    def train(self, input, output, epochs, showProgress=False):        
        if not len(input) == len(output):
            print("Number of training inputs does not match number of training outputs!")
            return

        for i in xrange(epochs):
            hiddenLayer, outputLayer = self.forward(input);
            hiddenDrift, outputDrift = self.drift(input, hiddenLayer, outputLayer, output)

            self.hiddenWeights += hiddenDrift
            self.outputWeights += outputDrift

            if showProgress:
                progress = (float(i + 1) / float(epochs)) * 100.0
                print("Learned: " + str(int(progress)) + "%")

#Get all files of a certain type in a specified folder
def get_filenames(dir):
    files = []
    files.extend(glob.glob1(dir, '*.fts'))
    return files

#Save the weights to a weight file
def save_weights(name, dir, weights):
    weights.dump(dir + "/" + name + ".weight")

#Load the weights from a certain directory
def load_weights(filedir):
    return np.load(filedir)



# returns name of class
def result_to_string(result):
    classes = [line.rstrip('\n') for line in open(os.path.realpath('classes.txt'))]
    maxIndex = result.index(max(result))
    outputName = classes[maxIndex]
    return outputName



if __name__ == "__main__":
    argc = len(sys.argv)
    np.set_printoptions(threshold=np.nan)

    input = []

    ready = True
    train = False

    epochs = 10000
    testInput = []
    weightsDir = ""

    #TODO: Laad een 2D matrix (zoals op regel 132 hardcoded) van outputs
    #TODO: Krijg je leaftype voor een bepaalde file(staat hier onder)
    def returnSoort(imagename):
        f = open(os.path.realpath('imageclef_testwithgroundtruthxml.csv'))
        csv_f = csv.reader(f)
        for row in csv_f:
            if row[0].find(imagename) > 0:
                return row.pop()
                break
    #TODO: Zet deze leaftype om naar een array [0, 0, 0, 0, 1, 0, 0] (staat hier onder)
    def inputNN(leaftype):
        classes = [line.rstrip('\n') for line in open(os.path.realpath('classes.txt'))]
        outputArray = []
        for i in xrange(0,len(classes)):
            outputArray.append(0)
        index = classes.index(leaftype)
        replacement = 1
        outputArray[index] = replacement
        return outputArray


    if (argc is 4 and sys.argv[1] == "-t"):
        dir = sys.argv[2]
        epochs = int(sys.argv[3])

        files = get_filenames(dir);
        for file in files:
            input.append(np.loadtxt(dir + '/' + file))
            #leafTypeString = load_leaftype(dir + '/' + file)
            #output.append(leaftype_to_array(leafTypeString))

        input = np.array(input)
        train = True
    elif (argc is 3):
        weightsDir = sys.argv[1]
        file = sys.argv[2]

        testInput.append(np.loadtxt(file))    
        testInput = np.array(testInput)            
    else:
        print('Warning: Number of arguments not recognized\nArgument format (ommit <>):\nTraining: -t <dir to train folder> <epochs>\nTesting: <dir to folder weight data> <dir to file>')
        ready = False

    if (ready):
        if (train):
            output = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1]]) #Deze regel weghalen als je de ouput kan laden

            nHiddenNodes = int(np.mean(len(input[0]) + len(output[0])))
            nn = NeuralNetwork(nHiddenNodes)
            nn.train(input, output, epochs, True)

            save_weights("hidden", dir, nn.hiddenWeights)
            save_weights("output", dir, nn.outputWeights)
        else:
            hiddenWeights = load_weights(weightsDir + "/hidden.weight")
            outputWeights = load_weights(weightsDir + "/output.weight")

            nn = NeuralNetwork(0, hiddenWeights, outputWeights)
            result = nn.forward(testInput)[1]
            leafType = result_to_string(result)

            print (result)
            print (leafType)
