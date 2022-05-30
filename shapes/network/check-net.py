import numpy as np
from matplotlib import pyplot as plt
import time

# ----------------------------------------
# Załadowanie danych treningowych
U = np.loadtxt("net-data/net-testData.csv", delimiter=',')
C = np.loadtxt("net-data/net-testLabels.csv", delimiter=',')
samplesNo = len(U)
# --------------------

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
# Współczynnik szybkości uczenia się
init_ro = 0.1
ro_decay = 0.0001
ro = init_ro
# ----------------------------------------
# Współczynnik metody momentum RHW
a = 0.9
# ----------------------------------------
# Parametry pętli
run = True
IleKrokow=0
# ----------------------------------------
Loss = np.zeros((samplesNo))
Accuracy = np.zeros((samplesNo))
Confidence = np.zeros((samplesNo))
# ----------------------------------------

while (run):
    ek = int(input("Provide image ID: "))
    
    t = time.process_time()
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
    Loss[IleKrokow] = loss
    predicted = np.argmax(l2_U)
    Accuracy[IleKrokow] = C[ek][predicted]
    Confidence[IleKrokow] = l2_U[predicted]
    acc = sum(Accuracy) / (IleKrokow+1)
    
    # ----------------------------------------
    # Skynet Says
    # ----------------------------------------
    print("True answer:",correct)
    print("SkyNET says:",predicted)
    elapsed_time = time.process_time() - t
    print("Response time: "+str(elapsed_time))
    # --------------------
    
    showImage = input("To show image type [s], or press enter to continue: ")
    if showImage == "s":
        plt.imshow(np.reshape(U[ek],(28,28)), cmap='gray',interpolation='nearest')
        plt.show()
    again = input("To finish type [q], or press enter to continue: ")
    if again == "q":
        run = False
    else:
        IleKrokow += 1
    
    # ----------------------------------------
    # ----------------------------------------

# --------------------
np.savetxt("net-data/net-model_User-test-Loss.csv", Loss, delimiter=',')
np.savetxt("net-data/net-model_User-test-Accuracy.csv", Accuracy, delimiter=',')
np.savetxt("net-data/net-model_User-test-Accuracy.csv", Confidence, delimiter=',')
# ----------------------------------------
Total_Loss = np.sum(Loss) / (IleKrokow+1)
Total_Accuracy = np.sum(Accuracy) / (IleKrokow+1)
print("Total loss:",Total_Loss,"Total accuracy:",Total_Accuracy)
# ----------------------------------------
# Rysowanie wykresów
fig, (ax1,ax2,ax3) = plt.subplots(3)
fig.suptitle("SoftMax test indywidualny")
# ----------
ax1.set_title("Krzywa błędu uczenia")
ax1.axis([0,IleKrokow,0,5])
ax1.plot(Loss, 'k')
# ----------
ax2.set_title("Celność")
ax2.axis([0,IleKrokow,0,1.1])
ax2.plot(Accuracy, 'k')
# ----------
ax3.set_title("Pewność")
ax3.axis([0,IleKrokow,0,1.1])
ax3.plot(Confidence, 'k')
# ----------
plt.show()
