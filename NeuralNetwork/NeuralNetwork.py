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
        ah = self.sigmoid(np.dot(X,model['w1']) + model['b1'])
        ao = self.sigmoid(np.dot(ah,model['w2']) + model['b2'])
        return ah, ao

    def train(self, X, y, n_hid, epochs, batch=1, learn=0.1):    
        n_in = len(X[0])
        n_out = len(y[0])            
        w1 = 2*np.random.random((n_in,n_hid)) - 1
        w2 = 2*np.random.random((n_hid,n_out)) - 1

        b1 = np.zeros((1, n_hid))
        b2 = np.zeros((1, n_out))
               
        model = { 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2} 
        indices = np.arange(n_in+1-(batch/2))

        for j in xrange(epochs):
            np.random.shuffle(indices)
            for i in indices:
                x = X[i*batch:i*batch+batch]
                tar_y = y[i*batch:i*batch+batch]
                if len(x) > 0:
                    ai = x
                    ah, ao = self.feedforward(ai, model)

                    ao_error = tar_y - ao
                    ao_delta = ao_error*self.dsigmoid(ao)

                    ah_error = ao_delta.dot(model['w2'].T)
                    ah_delta = ah_error * self.dsigmoid(ah)

                    dr2 = np.dot(ah.T, ao_delta)
                    dr1 = np.dot(ai.T, ah_delta)

                    dr2_bias = np.sum(ao_delta, axis=0)
                    dr1_bias = np.sum(ah_delta, axis = 0)

                    #reg_w2 += reg_lambda * w2
                    #reg_w1 += reg_lambda * w1

                    model['w2'] += learn * dr2
                    model['w1'] += learn * dr1

                    model['b2'] += learn * dr2_bias
                    model['b1'] += learn * dr1_bias     
        
            if (epochs >= 10 and j%(epochs/10)) == 0:
                print "Epoch:" + str(j + (epochs/10)) + "/" + str(epochs)+ " : " + "Error:" + str(np.sum(ao_error**2))
                
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
def result_to_string(dir, result):
	classes = [line.rstrip('\n') for line in open(dir + '/' + 'classes.txt')]
	result = result.tolist()
	#maxIndex = result[0].index(max(result[0]))
	#outputName = classes[maxIndex]
	sortres = np.sort(result[0])[:5]
	output = ["","","","",""]
	for i in range(5):
            output[i] = classes[result[0].index(sortres[i])]
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

def train_dir(traindir, testdir, n_hid, epochs, batch, learn):  
    absDir = os.path.abspath(os.path.dirname(__file__)) 
    input, output = build_data(traindir)

    if (len(input) > 0): 
        print "Building model..."
        nn = NeuralNetwork()     
        model = nn.train(X=input, y=output, n_hid=n_hid, epochs=epochs, batch=batch, learn=learn)
        save_weights(absDir, np.array(model['w1']), np.array(model['w2']), np.array(model['b1']), np.array(model['b2']))
        print "Testing model..."
        test_folder(testdir, nn, model)
    else:
        print "Error: Could not find images specified in image table"

def test_folder(dir, nn, model):
    absDir = os.path.abspath(os.path.dirname(__file__))
    fileNames = get_filenames(dir)

    total = 0
    correct = 0
    
    for file in fileNames:
        input = []
        input.append(np.loadtxt(dir + "/" + file))	
        input = np.array(input)

        answer = get_leaftype(absDir, os.path.splitext(file)[0])
        result = result_to_string(absDir, nn.predict(input, model))
        for i in range(5):            
            if (result[i] == answer):
                correct += 1
                print "res: " + result[i] + " | ans: " + answer
                break
        total += 1

    print("Finished with testing accuracy of " + str(float(correct) / float(total) * 100) + " %") 

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
        batch = int(sys.argv[5])
        learn = float(sys.argv[6])
        train_dir(traindir, testdir, n_hid, epochs, batch, learn)
    elif (argc is 2):
        dir = sys.argv[1]
        test_image(dir)
    else:
        print "Invalid argument stucture, the struture should be as follows (ommit <>):\nTraining: <dir train> <dir test> <hidden nodes> <epochs> <batches> <learn rate>\nTesting: <dir to file>"

    
