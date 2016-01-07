import csv
import os
import sys
#maakt van een planten- of boomsoort een array met nullen en een 1 waar het neural network mee om kan gaan
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

#vertaalt de output van het neural network naar een planten- of boomsoort
def outputNN(outputNNArray):
    classes = [line.rstrip('\n') for line in open('classes.txt')]
    maxIndex = outputNNArray.index(max(outputNNArray))
    outputName = classes[maxIndex]
    return outputName

print outputNN(inputNN("Acer negundo"))

#returned welke soort bij welke image hoort
def returnSoort(imagename):
    f = open("imageclef_testwithgroundtruthxml.csv")
    csv_f = csv.reader(f)
    path = os.path.realpath('..')
    print path
    for row in csv_f:
        if row[0].find(imagename) > 0:
            return row.pop()
            break

print returnSoort("10001.jpg")

