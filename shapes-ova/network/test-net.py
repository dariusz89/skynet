import numpy as np
from matplotlib import pyplot as plt
import time
import math
E = math.e
# ----------------------------------------
# Załadowanie danych testowych
U = np.loadtxt("net-data/net-testData.csv", delimiter=',')
C = np.loadtxt("net-data/net-testLabels.csv", delimiter=',')
if len(U) == len(C):
    print("Załadowano dane treningowe")
    print("Ilość wektorów trenujących: "+str(len(U)))
    print("Ilość wektorów klasyfikacyjnych: "+str(len(C)))
    samplesNo = len(U)
else:
    print("Nie poprawne rozmiary danych treningowych")
    exit()
# ----------------------------------------

# ----------------------------------------
def layer(A,W,B,BW):
    neuronsNumber = len(BW)
    S = np.zeros((neuronsNumber))
    for i in range(0,neuronsNumber):
        S[i] = BW[i] * B
        for j in range(0,len(A)):
            S[i] += W[i][j] * A[j]
    U = np.zeros((neuronsNumber))
    for i in range(0,neuronsNumber):
        U[i] = 1 / (1+np.exp(-S[i]))
    return S,U
# ----------------------------------------
bias = 1
# --------------------
# Warstwa 1
l1N = 4
l1_W  = np.zeros((l1N,784))
l1_BW = np.zeros((l1N))
l1_S = np.zeros((l1N))
l1_U = np.zeros((l1N))
# --------------------
# Warstwa 2
l2N = 1
l2_W  = np.zeros((l2N,l1N))
l2_BW = np.zeros((l2N))
l2_S = np.zeros((l2N))
l2_U = np.zeros((l2N))
# ----------------------------------------

# ----------------------------------------
# Zliczanie błędów do wykresu błędu uczenia
Error = np.zeros((samplesNo))
Accuracy = np.zeros((samplesNo))
Total_Error = 0
Total_Accuracy = 0
# ----------------------------------------
# Parametry pętli
Iteracja = 0
krok = 1
IleKrokow=samplesNo * krok
t_all = time.process_time()
# ----------------------------------------
while (Iteracja < IleKrokow):
    ek = np.random.randint(samplesNo)
    out = np.zeros((4))
    for category in range(0,4):
        l1_W  = np.loadtxt("net-data/net-model_"+str(category)+"-trained-l1_W.csv", delimiter=',', ndmin=2)
        l1_BW = np.loadtxt("net-data/net-model_"+str(category)+"-trained-l1_BW.csv", delimiter=',', ndmin=1)
        l2_W  = np.loadtxt("net-data/net-model_"+str(category)+"-trained-l2_W.csv", delimiter=',', ndmin=2)
        l2_BW = np.loadtxt("net-data/net-model_"+str(category)+"-trained-l2_BW.csv", delimiter=',', ndmin=1)
        l1_S,l1_U = layer(U[ek],l1_W,bias,l1_BW)
        l2_S,l2_U = layer(l1_U,l2_W,bias,l2_BW)
        out[category] = l2_U[0]
    print("#"+str(Iteracja),"out:",np.round(out,decimals=2))
    print("#"+str(Iteracja),"C:",C[ek])
    predicted = np.argmax(out)
    target = np.argmax(C[ek])
    Error[Iteracja] = abs(C[ek][target] - out[predicted])
    prediction = (out[predicted] > 0.5) * 1
    if C[ek][target] == prediction:
        Accuracy[Iteracja] = 1
    else:
        Accuracy[Iteracja] = 0
    #print("#"+str(Iteracja),"P:",prediction,"C:",int(C[ek][target]),"err:",Error[Iteracja])
    Iteracja += 1
elapsed_time = time.process_time() - t_all
print("Czas pochłonięty przez trening skynetu: "+str(elapsed_time))
np.savetxt("net-data/net-model_OvA-test-Error.csv", Error, delimiter=',')
Total_Error = np.sum(Error) / IleKrokow
Total_Accuracy = np.sum(Accuracy) / IleKrokow
print("Total error:",Total_Error,"Total accuracy:",Total_Accuracy)
# Rysowanie wykresów
fig, (ax1,ax2) = plt.subplots(2)
fig.suptitle("One vs All")
# ----------
ax1.set_title("Krzywa błędu uczenia")
ax1.axis([0,IleKrokow/krok,0,2.5])
ax1.plot(Error, 'k')
# ----------
ax2.set_title("Celność")
ax2.axis([0,IleKrokow/krok,0,2.5])
ax2.plot(Accuracy, 'k')
# ----------
plt.show()





