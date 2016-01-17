#!/usr/bin/env python

import sys
import os
import glob
import xml
from xml.dom import minidom

# Deletes all *.fts files containing SIFT features of 'photograph'-type images in a specified directory.
#
# Arguments:
# 	- directory (required)
#
# Example call: "python delete.py imageclef/train".
#
#


def main(argv):

	argc = len(argv)
	
	if (argc is not 1):
		print '\nWarning: incorrect argument(s) to knn.py. Expected arguments:\n\n' \
		'\t- directory (required)\t: directory containing  *.fts files with SIFT features.\n'
		sys.exit(1)
	
	dir = argv[0]
	
	xmls = []
	xmls.extend(glob.glob1(dir, '*.xml'))
	
	print '{} .xml files found in directory {}'.format(len(xmls), dir)
	
	deleted = 0
	
	for xml in xmls:
		xml1 = open(dir + '/' + xml)
		xml2 = minidom.parse(xml1)
		itemlist =  xml2.getElementsByTagName('Type')
		
		
		if itemlist[0].firstChild.nodeValue == 'photograph':
			
			deleted += 1
			fts = dir + '/' + xml.replace('.xml','.fts')
			# If text file exists, delete it.
			if (os.path.isfile(fts)):
				os.remove(fts)

	print '*.fts files of {} "photograph" and "pseudoscan" images deleted.'.format(deleted)
	
if __name__ == '__main__':
	main(sys.argv[1:])
