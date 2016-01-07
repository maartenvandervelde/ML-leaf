import numpy as np


def inputNN(soort):
    classes = [line.rstrip('\n') for line in open('classes.txt')]
    outputArray = []
    for i in xrange(0,len(classes)):
        outputArray.append(0)
    index = classes.index(soort)
    replacement = 1
    outputArray[index] = replacement
    return outputArray

print inputNN("Acer negundo")


def outputNN(outputNNArray):
    classes = [line.rstrip('\n') for line in open('classes.txt')]
    maxIndex = outputNNArray.index(max(outputNNArray))
    outputName = classes[maxIndex]
    return outputName

print outputNN(inputNN("Acer negundo"))