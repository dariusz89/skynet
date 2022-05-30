# ====================
# generate_sample_data.py
# ====================
# Autor: Dariusz Świąder
# --------------------
# Cel: przygotowanie danych do wytrenowania i przetestowania sieci
# --------------------
# Po przygotowaniu sieci i jej przetestowaniu nic nie będzie stało na przeszkodzie,
# aby zaimplementować istniejące już zbiory obrazków np.
# kształty:
# https://github.com/googlecreativelab/quickdraw-dataset
# ręcznie pisane cyfry, litery:
# http://yann.lecun.com/exdb/mnist/
# https://www.nist.gov/srd/nist-special-database-19
# ====================
from PIL import Image, ImageDraw
from random import seed
from random import randint
import numpy as np
import os
# --------------------
# Podstawowe kształty
# --------------------
def circle(imageSize,xy0,xy1):
	img = Image.new("RGB", imageSize, imageBackgroundColor)
	img1 = ImageDraw.Draw(img)
	shapeSize = [(xy0, xy0), (xy1, xy1)]
	if fillShape:
		img1.ellipse(shapeSize, fill=fillColor)
	elif outlineShape:
		img1.ellipse(shapeSize, outline=outlineColor)
	return img
def pacman(imageSize,xy0,xy1):
	img = Image.new("RGB", imageSize, imageBackgroundColor)
	img1 = ImageDraw.Draw(img)
	shapeSize = [(xy0, xy0), (xy1, xy1)]
	if fillShape:
		img1.pieslice(shapeSize, start = 30, end = 330, fill=fillColor)
	elif outlineShape:
		img1.pieslice(shapeSize, start = 30, end = 330, outline=outlineColor)
	return img
def square(imageSize,xy0,xy1):
	img = Image.new("RGB", imageSize, imageBackgroundColor)
	img1 = ImageDraw.Draw(img)
	shapeSize = [(xy0, xy0), (xy1, xy1)]
	if fillShape:
		img1.rectangle(shapeSize, fill=fillColor)
	elif outlineShape:
		img1.rectangle(shapeSize, outline=outlineColor)
	return img
def rectangle(imageSize,xy0,xy1):
	img = Image.new("RGB", imageSize, imageBackgroundColor)
	img1 = ImageDraw.Draw(img)
	shapeSize = [(xy0, xy0), (xy1, xy1*2-1)]
	if fillShape:
		img1.rectangle(shapeSize, fill=fillColor)
	elif outlineShape:
		img1.rectangle(shapeSize, outline=outlineColor)
	return img
def triangle(imageSize,xy0,xy1):
	if (xy0 % 2) != 0:
		xy0 = xy0 - 1
	if (xy1 % 2) != 0:
		xy1 = xy1 - 1
	img = Image.new("RGB", imageSize, imageBackgroundColor)
	img1 = ImageDraw.Draw(img)
	shapeSize = [(xy0,xy1), (xy1,xy1), (xy1, xy0)]
	if fillShape:
		img1.polygon(shapeSize, fill=fillColor)
	elif outlineShape:
		img1.polygon(shapeSize, outline=outlineColor)
	return img
# --------------------
# Generowanie n-obrazków dla poniższych kategorii:
# - pacman
# - circle
# - square
# - triangle
# - rectangle
# --------------------
def randomPacman(shapeMinSize,numberOfImages):
	pacmans = []
	for i in range(numberOfImages):
		xy0 = 0
		xy1 = 0
		rotate = [0,90,180,270]
		while ((xy1-xy0) < shapeMinSize):
			xy0 = randint(1, 8)
			xy1 = randint(20, 27)
		img = pacman(imageSize,xy0,xy1).rotate(rotate[randint(0, 3)])
		#img = pacman(imageSize,xy0,xy1)
		pacmans.append(img)
	return pacmans
def randomCircle(shapeMinSize,numberOfImages):
	circles = []
	for i in range(numberOfImages):
		xy0 = 0
		xy1 = 0
		rotate = [0,90,180,270]
		while ((xy1-xy0) < shapeMinSize):
			xy0 = randint(1, 8)
			xy1 = randint(20, 27)
		img = circle(imageSize,xy0,xy1).rotate(rotate[randint(0, 3)])
		#img = circle(imageSize,xy0,xy1)
		circles.append(img)
	return circles
def randomSquare(shapeMinSize,numberOfImages):
	squares = []
	for i in range(numberOfImages):
		xy0 = 0
		xy1 = 0
		rotate = [0,90,180,270]
		while ((xy1-xy0) < shapeMinSize):
			xy0 = randint(1, 8)
			xy1 = randint(20, 27)
		img = square(imageSize,xy0,xy1).rotate(rotate[randint(0, 3)])
		#img = square(imageSize,xy0,xy1)
		squares.append(img)
	return squares
def randomTriangle(shapeMinSize,numberOfImages):
	triangles = []
	for i in range(numberOfImages):
		xy0 = 1
		xy1 = 0
		rotate = [0,90,180,270]
		while ((xy1-xy0) < shapeMinSize):
			xy0 = randint(1, 8)
			xy1 = randint(20, 27)
		img = triangle(imageSize,xy0,xy1).rotate(rotate[randint(0, 3)])
		#img = triangle(imageSize,xy0,xy1)
		triangles.append(img)
	return triangles
# --------------------
# Zapis obrazków, oraz datasetów zawierających tablice z obrazkami:
# - obrazki
# - tablice NumPy
# Wymiary tablicy: numberOfImages x size x size
# --------------------
def saveImages(imagesSets):
	for imagesSet in imagesSets:
		imgArray = imagesSet[0]
		numpyArr = imagesSet[1]
		filePath = imagesSet[2]
		fileName = imagesSet[3]
		for i in range(0,numberOfImages):
			if not os.path.exists(filePath):
				os.makedirs(filePath)
			imgArray[i].save(filePath+fileName+"_"+str(i)+".png")
			numpyArr[i] = np.asarray(imgArray[i].convert("L"),dtype=np.uint8)
		np.savez(("data/"+fileName+"_dataset"),numpyArr)
# --------------------
# Test odczytu wybranego obrazka z zapisanego datasetu
# --------------------
def testDataset(datasets,num):
	for dataset in datasets:
		test_dataset = np.load(dataset)
		print(test_dataset['arr_0'][num])
		Image.fromarray(test_dataset['arr_0'][num]).show()
# --------------------
# Parametry generatora obrazków:
# - numberOfImages       = ilość obrazków do wygenerowania
# - size                 = rozmiar obrazka odpowiadający za szerokość i wysokość
# - shapeMinSize         = minimalny rozmiar kształtu odpowiadający za szerokość i wysokość 
# --------------------
# - imageSize            = rozmiary obrazka
# - imageBackgroundColor = kolor tła
# - fillShape            = czy wypełniać kształt kolorem?
# - fillColor            = kolor
# - outlineShape         = czy obrysować kształt?
# - outlineColor         = kolor
# --------------------
# Inicjalizacja generator liczb losowych
# --------------------
# Przygotowanie zbiorów z obrazkami, oraz odpowiadających im datasetów i ich zapis
# --------------------
# Odczyt wybranego obrazka z każdego datasetu
# --------------------
numberOfImages = 1000
size = 28
shapeMinSize = 8
# --------------------
imageSize = (size,size)
imageBackgroundColor = "#000000"
fillShape = False
fillColor = "#FFFFFF"
outlineShape = True
outlineColor = "#FFFFFF"
# --------------------
seed(1)
# --------------------
imagesSets = (
	( randomPacman(shapeMinSize,numberOfImages), np.zeros((numberOfImages,size,size)), "data/pacmans/", "pacman" ),
	( randomCircle(shapeMinSize,numberOfImages), np.zeros((numberOfImages,size,size)), "data/circles/", "circle" ),
	( randomSquare(shapeMinSize,numberOfImages), np.zeros((numberOfImages,size,size)), "data/squares/", "square" ),
	( randomTriangle(shapeMinSize,numberOfImages), np.zeros((numberOfImages,size,size)), "data/triangles/", "triangle" )
)
saveImages(imagesSets)
# --------------------
#num = randint(0,numberOfImages)
#print("--------------------")
#print("Wybrany losowo numer obrazka: " + str(num))
#print("--------------------")
#datasets = ( "pacman_dataset.npz","circle_dataset.npz","square_dataset.npz","triangle_dataset.npz","rectangle_dataset.npz" )
#testDataset(datasets,num)
#print("--------------------")
# --------------------
