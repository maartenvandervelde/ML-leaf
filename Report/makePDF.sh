#!/bin/bash

pdflatex ML_leaf_report.tex
bibtex ML_leaf_report.aux
pdflatex ML_leaf_report.tex
pdflatex ML_leaf_report.tex
