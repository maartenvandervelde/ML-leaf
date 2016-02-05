# ML-leaf
A machine learning project to classify plant species based on images of their leaves.

## How to run
Note that the data set itself is not provided here but can be downloaded from http://otmedia.lirmm.fr/LifeCLEF/ImageCLEF2011-2012-2013/ImageCLEF2012PlantIdentificationTaskFinalPackage.zip.
- Run bagOfWords.py to perform feature extraction (e.g.: python bagOfWords.py imageclef/train imageclef/test 100 100).
- Run knn.py to classify with K-Nearest Neighbour (e.g.: python knn.py train test 100 5).
- Run NeuralNetwork/NeuralNetwork.py to train and test using the neural network (e.g.: python NeuralNetwork.py train test n_hidden_nodes epochs learn_rate reg; python NeuralNetwork.py testfile.hst)


## Dataset
This project uses the ImageCLEF 2012 Plant dataset (11,572 images of 126 species).
