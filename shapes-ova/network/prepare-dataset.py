import numpy as np

# ----------------------------------------
# Załadowanie danych treningowych
def loadData(No):
    print("----------------------------------------")
    print("Przygotowanie danych wejściowych skynetu:")
    print(" -> Rozpoczęto ładowanie zbiorów danych")
    pacman_dataset = np.load("data/pacman_dataset.npz")
    print("     -> data/pacman_dataset.npz    - OK")
    circle_dataset = np.load("data/circle_dataset.npz")
    print("     -> data/circle_dataset.npz    - OK")
    square_dataset = np.load("data/square_dataset.npz")
    print("     -> data/square_dataset.npz    - OK")
    triangle_dataset = np.load("data/triangle_dataset.npz")
    print("     -> data/triangle_dataset.npz  - OK")
    print(" -> Zakończono ładowanie zbiorów danych")
    print("    ------------------------------------")
    # --------------------
    print(" -> Rozpoczęto spłaszczanie zbiorów danych")
    pacmanFlatted = np.asarray([
    	np.reshape(pacman_dataset['arr_0'][i], -1)
    	for i in range(len(pacman_dataset['arr_0']))
    	],dtype=np.uint8)
    print("     -> Zbiór pacman               - OK")
    circleFlatted = np.asarray([
    	np.reshape(circle_dataset['arr_0'][i], -1)
    	for i in range(len(circle_dataset['arr_0']))
    	],dtype=np.uint8)
    print("     -> Zbiór circle               - OK")
    squareFlatted = np.asarray([
    	np.reshape(square_dataset ['arr_0'][i], -1)
    	for i in range(len(square_dataset ['arr_0']))
    	],dtype=np.uint8)
    print("     -> Zbiór square               - OK")
    triangleFlatted = np.asarray([
    	np.reshape(triangle_dataset['arr_0'][i], -1)
    	for i in range(len(triangle_dataset['arr_0']))
    	],dtype=np.uint8)
    print("     -> Zbiór triangle             - OK")
    print(" -> Zakończono spłaszczanie zbiorów danych")
    # --------------------
    print("----------------------------------------")
    print("Wyznaczenie wektorów treningowych i klasyfikacyjnych:")
    dataU = np.empty((No*4,784), dtype=np.uint8)
    classification = np.zeros((No*4,4),dtype=bool)
    for i in range(0,No):
        dataU[i] = (pacmanFlatted[i] > 127)
        classification[i,0] = True
    for i in range(0,No):
        dataU[i+No] = (circleFlatted[i] > 127)
        classification[i+No,1] = True
    for i in range(0,No):
        dataU[i+No*2] = (squareFlatted[i] > 127)
        classification[i+No*2,2] = True
    for i in range(0,No):
        dataU[i+No*3] = (triangleFlatted[i] > 127)
        classification[i+No*3,3] = True
    print(" -> pacman_dataset:    "+str(classification[0]))
    print(" -> circle_dataset:    "+str(classification[No]))
    print(" -> square_dataset:    "+str(classification[No*2]))
    print(" -> triangle_dataset:  "+str(classification[No*3]))
    print("----------------------------------------")
    print("Rozpoczęto tasowanie wektorów klasyfikacyjnych")
    print("wraz z odpowiadającymi im wektorami treningowymi:")
    randomize = np.arange(len(dataU))
    np.random.shuffle(randomize)
    print("Generowanie nowych indeksów")
    dataU = dataU[randomize]
    print(" -> Potasowano wektory treningowe")
    classification = classification[randomize]
    print(" -> Potasowano wektory klasyfikacyjne")
    print("----------------------------------------")
    print("Wyznaczenie zbiorów treningowych i testowych w stosunku 5:5")
    print("Łączna ilość wektorów w zbiorze danych: "+str(len(dataU)))
    trainSamples = int(len(dataU) * 5/10)
    trainSamplesStart = 0
    trainSamplesEnd = trainSamples
    trainData = dataU[trainSamplesStart:trainSamplesEnd]
    trainLabels = classification[trainSamplesStart:trainSamplesEnd]
    print(" -> Wyznaczono zbiór treningowy o rozmiarze: "+str(trainSamples))
    # --------------------
    testSamples = int(len(dataU) * 5/10)
    testSamplesStart = trainSamples
    testSamplesEnd = trainSamples + testSamples
    testData = dataU[testSamplesStart:testSamplesEnd]
    testLabels = classification[testSamplesStart:testSamplesEnd]
    print(" -> Wyznaczono zbiór testowy o rozmiarze: "+str(testSamples))
    print("Zakończono przygotowywanie zbiorów danych")
    print("----------------------------------------")
    return trainData, trainLabels, testData, testLabels

samplesNo = 1000
trainData,trainLabels,testData,testLabels = loadData(samplesNo)
np.savetxt("net-data/net-trainData.csv", trainData, delimiter=',')
np.savetxt("net-data/net-trainLabels.csv", trainLabels, delimiter=',')
np.savetxt("net-data/net-testData.csv", testData, delimiter=',')
np.savetxt("net-data/net-testLabels.csv", testLabels, delimiter=',')

#----------------------------------------
#Przygotowanie danych wejściowych skynetu:
# -> Rozpoczęto ładowanie zbiorów danych
#     -> data/pacman_dataset.npz    - OK
#     -> data/circle_dataset.npz    - OK
#     -> data/square_dataset.npz    - OK
#     -> data/triangle_dataset.npz  - OK
# -> Zakończono ładowanie zbiorów danych
#    ------------------------------------
# -> Rozpoczęto spłaszczanie zbiorów danych
#     -> Zbiór pacman               - OK
#     -> Zbiór circle               - OK
#     -> Zbiór square               - OK
#     -> Zbiór triangle             - OK
# -> Zakończono spłaszczanie zbiorów danych
#----------------------------------------
#Wyznaczenie wektorów treningowych i klasyfikacyjnych:
# -> pacman_dataset:    [ True False False False]
# -> circle_dataset:    [False  True False False]
# -> square_dataset:    [False False  True False]
# -> triangle_dataset:  [False False False  True]
#----------------------------------------
#Rozpoczęto tasowanie wektorów klasyfikacyjnych
#wraz z odpowiadającymi im wektorami treningowymi:
#Generowanie nowych indeksów
# -> Potasowano wektory treningowe
# -> Potasowano wektory klasyfikacyjne
#----------------------------------------
#Wyznaczenie zbiorów treningowych i testowych w stosunku 6:4
#Łączna ilość wektorów w zbiorze danych: 4000
# -> Wyznaczono zbiór treningowy o rozmiarze: 2400
# -> Wyznaczono zbiór testowy o rozmiarze: 1600
#Zakończono przygotowywanie zbiorów danych
#----------------------------------------

