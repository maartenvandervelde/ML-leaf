import sys
import os
import glob
import xml
from xml.dom import minidom

# Deletes all files that are not associated with the top 32 classes in the ImageCLEF2012 dataset from training/testing directories.


xmls = []
xmls =  glob.glob1(os.path.abspath(''),'*.xml')
top32classes = [
'Ulmus minor',
'Phillyrea angustifolia',
'Buxus sempervirens',
'Quercus ilex',
'Olea europaea',
'Viburnum tinus',
'Pittosporum tobira',
'Hedera helix',
'Ruscus aculeatus',
'Celtis australis',
'Cercis siliquastrum',
'Populus nigra',
'Nerium oleander',
'Euphorbia characias',
'Corylus avellana',
'Pistacia lentiscus',
'Acer campestre',
'Rhus coriaria',
'Acer monspessulanum',
'Cotinus coggygria',
'Crataegus monogyna',
'Juniperus oxycedrus',
'Ginkgo biloba',
'Daphne cneorum',
'Populus alba',
'Platanus x hispanica',
'Arbutus unedo',
'Fraxinus angustifolia',
'Robinia pseudoacacia',
'Laurus nobilis',
'Diospyros kaki',
'Punica granatum'
]

count = len(xmls)

for xml in xmls:
	xml1 = open(os.path.abspath(os.path.dirname(__file__)) + '/' + xml)
	xml2 = minidom.parse(xml1)
	itemlist =  xml2.getElementsByTagName('ClassId')
	if itemlist[0].firstChild.nodeValue not in top32classes:
		count -= 1
		jpg = xml.replace('.xml','.jpg')
		fts = xml.replace('.xml','.fts')
		hst = xml.replace('.xml','.hst') 
		if (os.path.isfile(jpg)):
			os.remove(os.path.abspath(os.path.dirname(__file__)) + '/' + jpg)
		if(os.path.isfile(fts)):
			os.remove(os.path.abspath(os.path.dirname(__file__)) + '/' + fts)
		if(os.path.isfile(hst)):
			os.remove(os.path.abspath(os.path.dirname(__file__)) + '/' + hst)

print '{} items deleted'.format(count)