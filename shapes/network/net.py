import numpy as np
from matplotlib import pyplot as plt
import time

# ----------------------------------------
# Załadowanie danych treningowych
U = np.loadtxt("net-data/net-trainData.csv", delimiter=',')
C = np.loadtxt("net-data/net-trainLabels.csv", delimiter=',')
if len(U) == len(C):
    print("Załadowano dane treningowe")
    print("Ilość wektorów trenujących: "+str(len(U)))
    print("Ilość wektorów klasyfikacyjnych: "+str(len(C)))
    samplesNo = len(U)
else:
    print("Nie poprawne rozmiary danych treningowych")
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
l1_Wd  = np.zeros((l1N,784))
l1_BW = np.zeros((l1N))
l1_BWd = np.zeros((l1N))
l1_S = np.zeros((l1N))
l1_U = np.zeros((l1N))
l1_F = np.zeros((l1N))
l1_d = np.zeros((l1N))
for j in range(0,l1N):
    for i in range(0,784):
        l1_W[j,i] = np.random.rand()-0.5
for i in range(0,l1N):
    l1_BW[i] = np.random.rand()-0.5
# --------------------
# Warstwa 2
l2N = 4
l2_W  = np.zeros((l2N,l1N))
l2_Wd  = np.zeros((l2N,l1N))
l2_BW = np.zeros((l2N))
l2_BWd = np.zeros((l2N))
l2_S = np.zeros((l2N))
l2_U = np.zeros((l2N))
l2_F = np.zeros((l2N))
l2_d = np.zeros((l2N))
for j in range(0,l2N):
    for i in range(0,l1N):
        l2_W[j,i] = np.random.rand()-0.5
for i in range(0,l2N):
    l2_BW[i] = np.random.rand()-0.5
# ----------------------------------------

# ----------------------------------------
# Współczynnik szybkości uczenia się
init_ro = 0.1
ro_decay = 0.0001
ro = init_ro
# ----------------------------------------
# Współczynnik metody momentum RHW
a = 0.9
# ----------------------------------------
# Parametry pętli
Iteracja = 0
krok = 10
IleKrokow=samplesNo*krok
# ----------------------------------------
# Dane diagnostyczne
Loss = np.zeros((samplesNo*krok))
Accuracy = np.zeros((samplesNo*krok))
Confidence = np.zeros((samplesNo*krok))
LearningRate = np.zeros((samplesNo*krok))
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
    if (Iteracja % krok) == 0:
        print("#"+str(Iteracja),"loss:",loss,"acc:",acc)
    # --------------------
    
    # ----------------------------------------
    # Faza propagacji wstecz
    # ----------------------------------------
    # Warstwa 2
    l2_dW = np.zeros((l2N,l1N))
    l2_dBW = np.zeros((l2N))
    l2_d = l2_U - C[ek]
    for i in range(0,len(l2_d)):
        l2_dW[i] += l2_d[i] * l1_U
    l2_dBW = l2_d
    # --------------------
    # Warstwa 1
    l1_dW = np.zeros((l1N,784))
    l1_dBW = np.zeros((l1N))
    l1_F = l1_U * (1 - l1_U)
    l1_d = np.dot(l2_d,l2_W) * l1_F
    for i in range(0,len(l1_d)):
        l1_dW[i] += l1_d[i] * U[ek]
    l1_dBW = l1_d
    # --------------------
    
    # ----------------------------------------
    # Aktualizacja współczynnika uczenia się
    # ----------------------------------------
    ro = init_ro * (1 / (1 + ro_decay * Iteracja))
    LearningRate[Iteracja] = ro
    # --------------------
    
    # ----------------------------------------
    # Aktualizacja wag
    # ----------------------------------------
    # Warstwa 2
    tmp_l2_W = l2_W
    l2_W -= ro * l2_dW + a * l2_Wd
    l2_Wd = l2_W - tmp_l2_W
    # ----------
    tmp_l2_BW = l2_BW
    l2_BW -= ro * l2_dBW + a * l2_BWd
    l2_BWd = l2_BW - tmp_l2_BW
    # --------------------
    # Warstwa 1
    tmp_l1_W = l1_W
    l1_W -= ro * l1_dW + a * l1_Wd
    l1_Wd = l1_W - tmp_l1_W
    # ----------
    tmp_l1_BW = l1_BW
    l1_BW -= ro * l1_dBW + a * l1_BWd
    l1_BWd = l1_BW - tmp_l1_BW
    # --------------------
    
    # ----------------------------------------
    Iteracja += 1
    # ----------------------------------------
# --------------------
elapsed_time = time.process_time() - t_all
print("Czas pochłonięty przez trening skynetu: "+str(elapsed_time))
# --------------------
np.savetxt("net-data/net-model_trained-l1_W.csv", l1_W, delimiter=',')
np.savetxt("net-data/net-model_trained-l1_BW.csv", l1_BW, delimiter=',')
np.savetxt("net-data/net-model_trained-l2_W.csv", l2_W, delimiter=',')
np.savetxt("net-data/net-model_trained-l2_BW.csv", l2_BW, delimiter=',')
np.savetxt("net-data/net-model_trained-Loss.csv", Loss, delimiter=',')
np.savetxt("net-data/net-model_trained-Accuracy.csv", Accuracy, delimiter=',')
np.savetxt("net-data/net-model_trained-Confidence.csv", Confidence, delimiter=',')
np.savetxt("net-data/net-model_trained-LearningRate.csv", LearningRate, delimiter=',')
# ----------------------------------------
# Rysowanie wykresów
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4)
fig.suptitle("SoftMax - trening")
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
ax4.set_title("Współczynnik uczenia")
ax4.axis([0,IleKrokow,0,0.12])
ax4.plot(LearningRate, ("#e91e63"))
# ----------
plt.show()
