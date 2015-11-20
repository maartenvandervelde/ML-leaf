#!/usr/bin/env python

## creates a CSV file of all imageCLEF images, with the columns image_path, image_type, and class.
## Run with the command: python imageclefCSVgenerator.py directory
## where 'directory' is the folder containing the xml files
## 20 Nov 2015 Maarten

import csv, glob, os, sys
import xml.etree.cElementTree as et

path = ""
if len(sys.argv) > 1:
	path = sys.argv[1] + "/" # directory path

# create list of all xml files in the directory
xmlFiles = glob.glob(path + '*.xml')

scans = []
pseudoscans = []
photos = []
plantTypes = []

# create CSV
with open(path + 'imageclef_' + os.path.basename(os.path.normpath(path)) + '.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	writer.writerow(['image_path'] + ['image_type'] + ['class'])
	

	# read image type and sort into list, and register plant type
	for f in xmlFiles:
	
		imgType = et.parse(f).find('Type').text
		plantType = et.parse(f).find('ClassId').text
	
		if plantType not in plantTypes:
			plantTypes.append(plantType)
	
		if imgType == 'Scan':
			scans.append(f)
		elif imgType == 'pseudoscan':
			pseudoscans.append(f)
		elif imgType == 'photograph':
			photos.append(f)
		else:
			print "Unexpected image type: " + imgType
		
		
		# add row to CSV
		imgPath = os.path.splitext(f)[0] + '.jpg'
		writer.writerow([imgPath] + [imgType] + [plantType])
	

# print information to terminal
allTypes = scans + pseudoscans + photos
print "There are %d images of %d plants in the folder '%s':\n%d scans, %d pseudoscans, and %d photographs" % (len(allTypes), len(plantTypes), path, len(scans), len(pseudoscans), len(photos))

print "Information successfully written to %s." % (path + 'imageclef.csv')
