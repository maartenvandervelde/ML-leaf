#!/usr/bin/env python

## Converts leafsnap-dataset-images.txt into a useable csv, with columns image_path, image_type, class.
## Run with the command: python leafsnapCSVgenerator.py directory
## where 'directory' is the folder containing the leafsnap-dataset-images.txt file.
## 20 Nov 2015 Maarten

import csv, sys

path = ""
if len(sys.argv) > 1:
	path = sys.argv[1] + "/" # directory path

lab = []
field = []
plantTypes = []

with open(path + 'leafsnap.csv', 'wb') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	writer.writerow(['image_path'] + ['image_type'] + ['class'])
	
	with open(path + 'leafsnap-dataset-images.txt') as leafsnap:
		next(leafsnap) # skip first line (header)
		
		for line in leafsnap:
			contents = line.split()
			imgPath = contents[1]
			imgType = contents[-1]
			plantType = ' '.join(contents[3:-1])
			
			if plantType not in plantTypes:
				plantTypes.append(plantType)
			
			if imgType == 'lab':
				lab.append(plantType)
			elif imgType == 'field':
				field.append(plantType)
			else:
				print "Unexpected image type: " + imgType
				
			# add row to CSV
			writer.writerow([imgPath] + [imgType] + [plantType])

# print information to terminal
allTypes = lab + field
print "There are %d images of %d plants in the folder '%s':\n%d lab, and %d field" % (len(allTypes), len(plantTypes), path, len(lab), len(field))

print "Information successfully written to %s." % (path + 'leafsnap.csv')

