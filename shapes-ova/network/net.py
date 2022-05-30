import numpy as np
from matplotlib import pyplot as plt
import time
import math
E = math.e
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
# ----------------------------------------

# ----------------------------------------
# Wybór kategorii do treningu klasyfikatora
category = 3
# ----------------------------------------
# #Wyświetlenie wybranego obrazka
# -> pacman_dataset:    [ 1 0 0 0 ]
# -> circle_dataset:    [ 0 1 0 0 ]
# -> square_dataset:    [ 0 0 1 0 ]
# -> triangle_dataset:  [ 0 0 0 1 ]
#imageNo = 8
#print(C[imageNo])
#plt.imshow(np.reshape(U[imageNo],(28,28)), cmap='gray',interpolation='nearest')
#plt.show()
#exit()
# ----------------------------------------

# ----------------------------------------
# Binarna klasyfikacja
# OvA => one vs all
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
def output_delta(U,C):
    F = 0
    F = U * (1 - U)
    d = 0
    d = (C - U) * F
    return F,d
# ----------------------------------------
def hidden_delta(U,W,Wid,d_next):
    F = 0
    F = U * (1 - U)
    d = 0
    for m in range(0,len(d_next)):
        d += W[m][Wid] * d_next[m]
    d = d * F
    return F,d
# ----------------------------------------
def update_weights(U,W,B,BW,ro,d,Wd,BWd,a):
    newBW = BW + a*BWd + ro * d * B
    BWd = newBW - BW
    newW = np.zeros((len(W)))
    for i in range(0,len(W)):
        newW[i] = W[i] + a*Wd[i] + ro * d * U[i]
        Wd[i] = newW[i] - W[i]
    return newBW,newW
# ----------------------------------------
bias = 1
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
l2N = 1
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
# Współczynnik szybkości uczenia się
init_ro = 0.1
ro_decay = 0.0001
# Współczynnik metody momentum RHW
a = 0.9
# ----------------------------------------
# Zliczanie błędów do wykresu błędu uczenia
Error = np.zeros((samplesNo))
err_id = 0
# ----------------------------------------
# Parametry pętli
Iteracja = 0
krok = 10
IleKrokow=samplesNo*krok
t_all = time.process_time()
t = time.process_time()
# ----------------------------------------
while (Iteracja < IleKrokow):
    ek = np.random.randint(samplesNo)
    # ----------------------------------------
    # Faza propagacji w przód
    # --------------------
    # Warstwa 1
    l1_S,l1_U = layer(U[ek],l1_W,bias,l1_BW)
    # --------------------
    # Warstwa 2
    l2_S,l2_U = layer(l1_U,l2_W,bias,l2_BW)
    # --------------------
    
    #print("#"+str(Iteracja),"ek:",ek,"out:",round(l2_U[0], 2),"C:",int(C[ek][category]))
    
    # ----------------------------------------
    # Faza propagacji wstecz
    # --------------------
    # Warstwa 2
    #l2_d = dsoftmaxloss(l2_U,C[ek])
    l2_F[0],l2_d[0] = output_delta(l2_U,C[ek][category])
    # --------------------
    # Warstwa 1
    for i in range(0,l1N):
        l1_F[i],l1_d[i] = hidden_delta(l1_U[i],l2_W,i,l2_d)
    # --------------------
    
    # ----------------------------------------
    # Aktualizacja współczynnika uczenia się
    # --------------------
    ro = init_ro * (1 / (1 + ro_decay * Iteracja))
    # --------------------
    
    # ----------------------------------------
    # Aktualizacja wag
    # --------------------
    # Warstwa 2
    for i in range(0,l2N):
        l2_BW[i],l2_W[i] = update_weights(l1_U,l2_W[i],bias,l2_BW[i],ro,l2_d[i],l2_Wd[i],l2_BWd[i],a)
    # --------------------
    # Warstwa 1
    for i in range(0,l1N):
        l1_BW[i],l1_W[i] = update_weights(U[ek],l1_W[i],bias,l1_BW[i],ro,l1_d[i],l1_Wd[i],l1_BWd[i],a)
    # --------------------
    
    # Wyznaczenie błędu uczenia
    if (Iteracja % krok) == 0:
        err_sum = 0
        for q in range(0,4):
            t_ek = np.random.randint(samplesNo)
            # Warstwa 1
            l1_S,l1_U = layer(U[t_ek],l1_W,bias,l1_BW)
            # --------------------
            # Warstwa 2
            l2_S,l2_U = layer(l1_U,l2_W,bias,l2_BW)
            # --------------------
            err_sum += abs(C[t_ek][category] - l2_U[0])
        Error[err_id] = err_sum
        if (Iteracja % krok) == 0:
            print("#"+str(Iteracja),"out:",round(l2_U[0], 2),"C:",int(C[t_ek][category]),"err:",Error[err_id])
        t = time.process_time()
        err_id += 1
    Iteracja += 1
elapsed_time = time.process_time() - t_all
print("Czas pochłonięty przez trening skynetu: "+str(elapsed_time))
np.savetxt("net-data/net-model_"+str(category)+"-trained-l1_W.csv", l1_W, delimiter=',')
np.savetxt("net-data/net-model_"+str(category)+"-trained-l1_BW.csv", l1_BW, delimiter=',')
np.savetxt("net-data/net-model_"+str(category)+"-trained-l2_W.csv", l2_W, delimiter=',')
np.savetxt("net-data/net-model_"+str(category)+"-trained-l2_BW.csv", l2_BW, delimiter=',')
np.savetxt("net-data/net-model_"+str(category)+"-trained-Error.csv", Error, delimiter=',')

# Rysowanie wykresów
fig, (ax1) = plt.subplots(1)
fig.suptitle("Klasyfikator: "+str(category))
# ----------
ax1.set_title("Krzywa błędu uczenia")
ax1.axis([0,IleKrokow/krok,0,2.5])
ax1.plot(Error, 'k')
# ----------
plt.show()





