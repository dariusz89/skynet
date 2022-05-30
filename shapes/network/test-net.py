import numpy as np
from matplotlib import pyplot as plt
import time

# ----------------------------------------
# Załadowanie danych treningowych
U = np.loadtxt("net-data/net-testData.csv", delimiter=',')
C = np.loadtxt("net-data/net-testLabels.csv", delimiter=',')
if len(U) == len(C):
    print("Załadowano dane testowe")
    print("Ilość wektorów testowych: "+str(len(U)))
    print("Ilość wektorów klasyfikacyjnych: "+str(len(C)))
    samplesNo = len(U)
else:
    print("Nie poprawne rozmiary danych testowych")
    exit()
# --------------------
#imageNo = 0
#print(C[imageNo])
#plt.imshow(np.reshape(U[imageNo],(28,28)), cmap='gray',interpolation='nearest')
#plt.show()
#exit()
# --------------------
# Warstwa 1
l1N = 4
l1_W  = np.zeros((l1N,784))
l1_BW = np.zeros((l1N))
l1_S = np.zeros((l1N))
l1_U = np.zeros((l1N))
# --------------------
# Warstwa 2
l2N = 4
l2_W  = np.zeros((l2N,l1N))
l2_BW = np.zeros((l2N))
l2_S = np.zeros((l2N))
l2_U = np.zeros((l2N))
# ----------------------------------------
l1_W  = np.loadtxt("net-data/net-model_trained-l1_W.csv", delimiter=',', ndmin=2)
l1_BW = np.loadtxt("net-data/net-model_trained-l1_BW.csv", delimiter=',', ndmin=1)
l2_W  = np.loadtxt("net-data/net-model_trained-l2_W.csv", delimiter=',', ndmin=2)
l2_BW = np.loadtxt("net-data/net-model_trained-l2_BW.csv", delimiter=',', ndmin=1)
# ----------------------------------------
# Parametry pętli
Iteracja = 0
IleKrokow=samplesNo
# ----------------------------------------
# Dane diagnostyczne
Loss = np.zeros((samplesNo))
Accuracy = np.zeros((samplesNo))
Confidence = np.zeros((samplesNo))
LearningRate = np.zeros((samplesNo))
# ----------------------------------------
# Czas treningu
t_all = time.process_time()
# ----------------------------------------
while (Iteracja < IleKrokow):
    ek = np.random.randint(samplesNo)
    # ----------------------------------------
    # Faza propagacji w przód
    # ----------------------------------------
    # Warstwa 1
    l1_S = np.dot(l1_W,U[ek]) + l1_BW
    l1_U = 1 / (1+np.exp(-l1_S))
    # --------------------
    # Warstwa 2
    l2_S = np.dot(l2_W,l1_U) + l2_BW
    exp_values = np.exp(l2_S - np.max(l2_S))
    l2_U = exp_values/np.sum(exp_values)
    # --------------------
    
    # ----------------------------------------
    # Loss
    # ----------------------------------------
    l2_U_clipped = np.clip(l2_U, 1e-7, 1 - 1e-7)
    correct = np.argmax(C[ek])
    loss = -np.log(l2_U_clipped[correct])
    # --------------------
    
    # ----------------------------------------
    # Dane diagnostyczne
    # ----------------------------------------
    Loss[Iteracja] = loss
    prediction = np.argmax(l2_U)
    Accuracy[Iteracja] = C[ek][prediction]
    Confidence[Iteracja] = l2_U[prediction]
    acc = 0
    if Iteracja > 0:
        acc = sum(Accuracy) / (Iteracja+1)
    if (Iteracja % 10) == 0:
        print("#"+str(Iteracja),"loss:",loss,"acc:",acc)
    # --------------------
    
    # ----------------------------------------
    Iteracja += 1
    # ----------------------------------------
# --------------------
elapsed_time = time.process_time() - t_all
print("Czas pochłonięty przez trening skynetu: "+str(elapsed_time))
# --------------------
np.savetxt("net-data/net-model_tested-Loss.csv", Loss, delimiter=',')
np.savetxt("net-data/net-model_tested-Accuracy.csv", Accuracy, delimiter=',')
np.savetxt("net-data/net-model_tested-Confidence.csv", Confidence, delimiter=',')
np.savetxt("net-data/net-model_tested-LearningRate.csv", LearningRate, delimiter=',')
# ----------------------------------------
# Rysowanie wykresów
fig, (ax1,ax2,ax3) = plt.subplots(3)
fig.suptitle("SoftMax - test")
# ----------
ax1.set_title("Krzywa błędu uczenia")
ax1.axis([0,IleKrokow,0,5])
ax1.plot(Loss, ("#2196f3"))
# ----------
ax2.set_title("Celność")
ax2.axis([0,IleKrokow,0,1.2])
ax2.plot(Accuracy, "#43a047")
# ----------
ax3.set_title("Pewność")
ax3.axis([0,IleKrokow,0,1.2])
ax3.plot(Confidence, ("#ffa000"))
# ----------
plt.show()
