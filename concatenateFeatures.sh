#!/bin/bash
# Example call: ./concatenateFeatures.sh 1000
# Concatenates the contents from the first N .txt files in the directory.


FILES=(*.txt)
FILENAME="features"$1".fts"

rm -f $FILENAME

echo "Combining features from the first $1 files into $FILENAME"
awk '{print}' ${FILES[@]:0:$1} | tr "\n" " " >  $FILENAME