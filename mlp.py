#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Multi-layer perceptron
# Copyright (C) 2011  Nicolas P. Rougier
#
# Distributed under the terms of the BSD License.
# -----------------------------------------------------------------------------
# This is an implementation of the multi-layer perceptron with retropropagation
# learning.
#
# Adapted from http://www.labri.fr/perso/nrougier/downloads/mlp.py by Maarten
# -----------------------------------------------------------------------------

import sys
import numpy as np


def sigmoid(x):
    ''' Sigmoid like function using tanh '''
    return np.tanh(x)

def dsigmoid(x):
    ''' Derivative of sigmoid above '''
    return 1.0-x**2

class MLP:
    ''' Multi-layer perceptron class. '''

    def __init__(self, *args):
        ''' Initialization of the perceptron with given sizes.  '''

        self.shape = args
        n = len(args)

        # Build layers
        self.layers = []
        # Input layer (+1 unit for bias)
        self.layers.append(np.ones(self.shape[0]+1))
        # Hidden layer(s) + output layer
        for i in range(1,n):
            self.layers.append(np.ones(self.shape[i]))

        # Build weights matrix (randomly between -0.25 and +0.25)
        self.weights = []
        for i in range(n-1):
            self.weights.append(np.zeros((self.layers[i].size,
                                         self.layers[i+1].size)))

        # dw will hold last change in weights (for momentum)
        self.dw = [0,]*len(self.weights)

        # Reset weights
        self.reset()

    def reset(self):
        ''' Reset weights '''

        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size,self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        ''' Propagate data from input layer to output layer. '''

        # Set input layer
        self.layers[0][0:-1] = data

        # Propagate from layer 0 to layer n-1 using sigmoid as activation function
        for i in range(1,len(self.shape)):
            # Propagate activity
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],self.weights[i-1]))

        # Return output
        return self.layers[-1]


    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        ''' Back propagate error related to target using lrate. '''

        deltas = []

        # Compute error on output layer
        error = target - self.layers[-1]
        delta = error*dsigmoid(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape)-2,0,-1):
            delta = np.dot(deltas[0],self.weights[i].T)*dsigmoid(self.layers[i])
            deltas.insert(0,delta)
            
        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T,delta)
            self.weights[i] += lrate*dw + momentum*self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error**2).sum()

    def __repr__(self):
        n_inputs = str(self.shape[0])
        hidden_layers = self.shape[1:-1]
        n_outputs =  str(self.shape[-1])
        return("Hi, I am an MLP with " + n_inputs + " input(s), " + str(len(hidden_layers)) + " hidden layer(s): " +  str(hidden_layers) + ", and " + n_outputs + " output node(s).")


# -----------------------------------------------------------------------------
def main(argv):

    import matplotlib
    import matplotlib.pyplot as plt

    def learn(network,samples, max_epochs=50000, min_error=0.0001, lrate=.1, momentum=0.1):
        
        error = [None] * max_epochs
        
        # Train 
        for i in range(max_epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            error[i] = network.propagate_backward( samples['output'][n], lrate, momentum )
            if (error[i] < min_error):
                print "Error below threshold of %f after %d epochs" % (min_error, i)
                break
            if (i == max_epochs-1):
                print "Network did not converge within %d epochs" % (max_epochs) 
        # Test
        for i in range(samples.size):
            output = network.propagate_forward( samples['input'][i] )
            print "input:", samples['input'][i]
            print "output:", ["{0:0.2f}".format(o) for o in output]
            print "expected:", str(samples['output'][i]), "\n"
        print
        
        # Plot error
        plt.plot(error)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.show()

        
    # -------------------------------------------------------------------------        
    # Define number and configuration of nodes.
    # TODO: use user-provided arguments
    inputs = 6
    outputs = 6
    hidden_layers = [20, 20]
    
    network = MLP(inputs, *(hidden_layers + [outputs]))
    print(network)
    
    # -------------------------------------------------------------------------
    print "Learning our own random function"
    network.reset()
    samples = np.zeros(inputs + outputs, dtype=[('input',  float, inputs), ('output', float, outputs)])

    # Generate random input-output pairs to learn
    for i in xrange(len(samples)):
        samples[i] = np.random.randint(2, size = inputs), np.random.randint(2, size = outputs)
    
    learn(network, samples)

if __name__ == '__main__':
    main(sys.argv)
