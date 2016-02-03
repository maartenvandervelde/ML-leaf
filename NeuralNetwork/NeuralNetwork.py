import numpy as np
import sys
import glob
import csv
import os

class NeuralNetwork(object):
    def __init__(self):
        np.random.seed(1)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def dsigmoid(self, y):
        return y*(1-y)

    def feedforward(self, X, model):
        ah = self.sigmoid(np.dot(X,model['w1'] + model['b1']))
        ao = self.sigmoid(np.dot(ah,model['w2'] + model['b2']))
        return ah, ao

    def train(self, X, y, n_hid, epochs, learn=0.1, reg_lambda=0.0):    
        n_in = len(X[0])
        n_out = len(y[0])
        
        w1 = 2*np.random.random((n_in,n_hid)) - 1
        w2 = 2*np.random.random((n_hid,n_out)) - 1

        b1 = np.zeros((1, n_hid))
        b2 = np.zeros((1, n_out))
               
        model = { 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2} 
        indices = np.arange(len(X))

        for j in xrange(epochs):            
            np.random.shuffle(indices)
            for i in indices:
                x = np.array([X[i]])
                tar_y = np.array([y[i]])
                
                hidden_layer, output_layer = self.feedforward(x, model)
                
                output_error = output_layer - tar_y
                output_delta = output_error * self.dsigmoid(output_layer)
                hidden_error = output_delta.dot(w2.T)
                hidden_delta = hidden_error * self.dsigmoid(hidden_layer)

                output_drift = hidden_layer.T.dot(output_delta)
                hidden_drift = x.T.dot(hidden_delta)
                bias_2_drift = np.sum(output_delta, axis=0)
                bias_1_drift = np.sum(hidden_delta, axis=0) 
                
                output_drift += reg_lambda * w2
                hidden_drift += reg_lambda * w1
                
                w2 -= learn * output_drift
                w1 -= learn * hidden_drift
                b2 -= learn * bias_2_drift
                b1 -= learn * bias_1_drift
        
                model = { 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
                
            if (epochs >= 10 and j%(epochs/10)) == 0:
                print "Epoch:" + str(j + (epochs/10)) + "/" + str(epochs)+ " : " + "Error:" + str(np.sum(output_error**2))
                
        return model

    def predict(self, x, model):
        return self.feedforward(x, model)[1]


    #Get all files of a certain type in a specified folder
def get_filenames(dir):
	files = []
	files.extend(glob.glob1(dir, '*.hst'))
	return files

#Save the weights to a weight file
def save_weights(dir, w1, w2, b1, b2):
    w1.dump(dir + "/w1" + ".weight")
    w2.dump(dir + "/w2" + ".weight")
    b1.dump(dir + "/b1" + ".weight")
    b2.dump(dir + "/b2" + ".weight")

#Load the weights from a certain directory
def load_weights(filedir):
	return np.load(filedir + "/w1.weight"), np.load(filedir + "/w2.weight"), np.load(filedir + "/b1.weight"), np.load(filedir + "/b2.weight")

#Returns name of class
def result_to_string(dir, result, top=5):
	classes = [line.rstrip('\n') for line in open(dir + '/' + 'classes.txt')]
	result = result.tolist()
	sortres = np.sort(result)[::-1]
	output = []
	for i in range(top):
            output.append(classes[result.index(sortres[i])])
	return output

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

def build_data(traindir):
    print "Loading data..."
    absDir = os.path.abspath(os.path.dirname(__file__))
    fileNames = get_filenames(traindir)

    input = []
    output = []

    for file in fileNames:
        leafTypeString = get_leaftype(absDir, os.path.splitext(file)[0])
        if (not leafTypeString is None):
            input.append(build_input(traindir, file))
            output.append(build_output(absDir, leafTypeString))

    return np.array(input), np.array(output)

def train_dir(traindir, testdir, n_hid, epochs, learn, reg):  
    absDir = os.path.abspath(os.path.dirname(__file__)) 
    input, output = build_data(traindir)

    if (len(input) > 0): 
        print "Building model..."
        nn = NeuralNetwork()     
        model = nn.train(input, output, n_hid, epochs, learn, reg)
        save_weights(absDir, np.array(model['w1']), np.array(model['w2']), np.array(model['b1']), np.array(model['b2']))
        print "Testing model..."
        test_folder(testdir, nn, model)
    else:
        print "Error: Could not find images specified in image table"

def test_folder(dir, nn, model, top=5):
    absDir = os.path.abspath(os.path.dirname(__file__))
    fileNames = get_filenames(dir)

    correct = 0
    for string in fileNames:
        input = np.array(np.loadtxt(dir + "/" + string))
        answer = get_leaftype(absDir, os.path.splitext(string)[0])
        result = result_to_string(absDir, nn.predict(input, model), top)
        for i in range(len(result)):
            if (result[i] == answer):
                 correct += 1

    print("Finished with testing accuracy of " + str(float(correct) / float(len(fileNames)) * 100) + " %") 
    print(str(correct) + " correct out of " + str(len(fileNames)))

def test_image(dir):
    absDir = os.path.abspath(os.path.dirname(__file__))
    input = []
    input.append(np.loadtxt(dir))	
    input = np.array(input)	
    
    model = {}
    model['w1'], model['w2'], model['b1'], model['b2'] = load_weights(absDir)

    nn = NeuralNetwork()
    result = nn.predict(input, model)
    leafType = result_to_string(absDir, result)
    print(result)
    print(leafType)	

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    np.seterr(all='ignore')
    argc = len(sys.argv)

    if (argc is 7):
        traindir = sys.argv[1]
        testdir = sys.argv[2]
        n_hid = int(sys.argv[3])
        epochs = int(sys.argv[4])
        learn = float(sys.argv[5])
        reg = float(sys.argv[6])
        train_dir(traindir, testdir, n_hid, epochs, learn, reg)
    elif (argc is 2):
        dir = sys.argv[1]
        test_image(dir)
    else:
        print "Invalid argument stucture, the struture should be as follows (ommit <>):\nTraining: <dir train> <dir test> <hidden nodes> <epochs> <batches> <learn rate>\nTesting: <dir to file>"

    
