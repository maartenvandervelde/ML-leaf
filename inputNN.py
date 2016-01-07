import csv

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

def returnSoort(imagename):
    f = open("imageclef_testwithgroundtruthxml.csv")
    csv_f = csv.reader(f)
    for row in csv_f:
        if row[0].find(imagename) > 0:
            return row.pop()
            break

print returnSoort("10001.jpg")
