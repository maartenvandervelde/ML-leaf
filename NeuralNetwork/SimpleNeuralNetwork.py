import numpy as np
import sys
import glob
import csv
import os

#The class that constructs a neural network
class NeuralNetwork():
    def __init__(self, input=None, output=None, hiddenWeights=None, outputWeights=None, seed=1):
		#Seed the random function (good practice for back checking)
        np.random.seed(seed)
        #See if we use a precalculated set of weights, if not initialize random weights
        if (hiddenWeights is None and outputWeights is None):
            nHiddenNodes = 100
            self.hiddenWeights = 2 * np.random.random((len(input[0]), nHiddenNodes)) - 1
            self.outputWeights = 2 * np.random.random((nHiddenNodes, len(output[0]))) - 1
        else:
            self.hiddenWeights = hiddenWeights
            self.outputWeights = outputWeights
    
    #The sigmoid function used for forward propagation to calculate the
    #different layers within the network
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        #return np.arctan(x)

	#The derivative of the sigmoid used to calculate the delta
    def sigmoidDerivative(self, x):
        return x * (1 - x)
        #return 1 / (x**2 + 1)

	#The forward propagation part, calculates the hidden and ouput layers
    def forward(self, x):
        hiddenLayer = self.sigmoid(np.dot(x, self.hiddenWeights))
        outputLayer = self.sigmoid(np.dot(hiddenLayer, self.outputWeights))
        return hiddenLayer, outputLayer

	#Calculate the delta error usefull for the backpropagation
    def delta(self, error, output):
        return error * self.sigmoidDerivative(output)

	#Calculate the amount that the weights need to change (here denoted as
	#"drift")
    def drift(self, input, hidden, output, targetOutput):
        outputError = targetOutput - output
        outputDelta = self.delta(outputError, output)
        outputDrift = np.dot(hidden.T, outputDelta)

        hiddenError = np.dot(outputDelta, self.outputWeights.T)
        hiddenDelta = self.delta(hiddenError, hidden)
        hiddenDrift = np.dot(input.T, hiddenDelta)
        return hiddenDrift, outputDrift, outputError

	#Combines the above functions to calculate the final weights
    def train(self, input, output, epochs, showProgress=False):		
        if not len(input) == len(output):
            print("Number of training inputs does not match number of training outputs!")
            return

        for i in xrange(epochs):
            hiddenLayer, outputLayer = self.forward(input)
            hiddenDrift, outputDrift, outputError = self.drift(input, hiddenLayer, outputLayer, output)

            self.hiddenWeights += hiddenDrift
            self.outputWeights += outputDrift

            if showProgress:
                progress = (float(i + 1) / float(epochs)) * 100.0
                print("Learned: " + str(int(progress)) + "%")
        print("Finished with a training error of " + str(np.mean(outputError) * 100) + " %")

#Get all files of a certain type in a specified folder
def get_filenames(dir):
	files = []
	files.extend(glob.glob1(dir, '*.fts'))
	return files

#Save the weights to a weight file
def save_weights(dir, hiddenWeights, outputWeights):
    hiddenWeights.dump(dir + "/hidden" + ".weight")
    outputWeights.dump(dir + "/output" + ".weight")

#Load the weights from a certain directory
def load_weights(filedir):
	return np.load(filedir + "/hidden.weight"), np.load(filedir + "/output.weight")

#Returns name of class
def result_to_string(dir, result):
	classes = [line.rstrip('\n') for line in open(dir + '/' + 'classes.txt')]
	result = result.tolist()
	maxIndex = result[0].index(max(result[0]))
	outputName = classes[maxIndex]
	return outputName

def get_leaftype(dir, imagename):
		f = open(dir + '/' + 'imagetable.csv')
		csv_f = csv.reader(f)
		for row in csv_f:
			if row[0] == imagename:
				return row.pop()

def build_input(dir, file):
    return np.loadtxt(dir + '/' + file)

def build_output(dir, leaftype):
	classes = [line.rstrip('\n') for line in open(dir + '/' + 'classes.txt')]
	outputArray = []
	for i in xrange(0,len(classes)):
		outputArray.append(0)
	index = classes.index(leaftype)
	replacement = 1
	outputArray[index] = replacement
	outputArray = np.array(outputArray)
	return outputArray

def train_dir(traindir, testdir, epochs):
    absDir = os.path.abspath(os.path.dirname(__file__))
    fileNames = get_filenames(traindir)

    input = []
    output = []

    for file in fileNames:
        leafTypeString = get_leaftype(absDir, os.path.splitext(file)[0]) 
        if (not leafTypeString is None):
            input.append(build_input(traindir, file))
            output.append(build_output(absDir, leafTypeString))

    input = np.array(input) 

    if (input.size and output):
        nn = NeuralNetwork(input, output)
        nn.train(input, output, epochs, True)
        save_weights(absDir, nn.hiddenWeights, nn.outputWeights)
        test_folder(testdir)
    else:
        print "Error: Could not find images specified in image table"

def test_folder(dir):
    absDir = os.path.abspath(os.path.dirname(__file__))
    fileNames = get_filenames(dir)

    hiddenWeights, outputWeights = load_weights(absDir)

    nn = NeuralNetwork(None, None, hiddenWeights, outputWeights)
    
    total = len(fileNames)
    correct = 0
    
    for file in fileNames:
        input = []
        input.append(np.loadtxt(dir + "/" + file))	
        input = np.array(input)	
        result = result_to_string(absDir, nn.forward(input)[1])
        answer = get_leaftype(absDir, os.path.splitext(file)[0])

        if (result == answer):
            correct += 1

    print("Finished with testing accuracy of " + str(float(correct) / float(total) * 100) + " %") 

def test_image(dir):
    absDir = os.path.abspath(os.path.dirname(__file__))
    input = []
    input.append(np.loadtxt(dir))	
    input = np.array(input)	
    
    hiddenWeights, outputWeights = load_weights(absDir)

    nn = NeuralNetwork(None, None, hiddenWeights, outputWeights)
    result = nn.forward(input)[1]
    leafType = result_to_string(absDir, result)
    print(result)
    print(leafType)	

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    np.seterr(all='ignore')
    argc = len(sys.argv)

    if (argc is 4):
        traindir = sys.argv[1]
        testdir = sys.argv[2]
        epochs = int(sys.argv[3])
        train_dir(traindir, testdir, epochs)
    elif (argc is 2):
        dir = sys.argv[1]
        test_image(dir)
    else:
        print "Invalid argument stucture, the struture should be as follows (ommit <>):\nTraining: <dir of folder> <epochs>\nTesting: <dir to file>"
