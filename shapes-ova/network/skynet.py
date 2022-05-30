import numpy as np
# ----------------------------------------
treshold = 0.5
# --------------------
# Warstwa 1
l1N = 4
l1_S = np.zeros((l1N))
l1_U = np.zeros((l1N))
# --------------------
# Warstwa 2
l2N = 1
l2_S = np.zeros((l2N))
l2_U = np.zeros((l2N))
# ----------------------------------------
model_l1_W = np.zeros((4,l1N,784))
model_l1_BW = np.zeros((4,l1N))
model_l2_W = np.zeros((4,l2N,l1N))
model_l2_BW = np.zeros((4,l2N))
for category in range(0,4):
    model_l1_W[category]  = np.loadtxt("network/net-data/net-model_"+str(category)+"-trained-l1_W.csv", delimiter=',', ndmin=2)
    model_l1_BW[category] = np.loadtxt("network/net-data/net-model_"+str(category)+"-trained-l1_BW.csv", delimiter=',', ndmin=1)
    model_l2_W[category]  = np.loadtxt("network/net-data/net-model_"+str(category)+"-trained-l2_W.csv", delimiter=',', ndmin=2)
    model_l2_BW[category] = np.loadtxt("network/net-data/net-model_"+str(category)+"-trained-l2_BW.csv", delimiter=',', ndmin=1)
def layer(A,W,BW):
    neuronsNumber = len(BW)
    S = np.zeros((neuronsNumber))
    #for i in range(0,neuronsNumber):
    #    S[i] = np.dot(W[i],A) + BW[i]*B
    S = np.dot(W,A) + BW
    U = np.zeros((neuronsNumber))
    U = 1 / (1 + np.exp(-S))
    return U
# ----------------------------------------
def net(U):
    out = np.zeros((4))
    for category in range(0,4):
        l1_W  = model_l1_W[category]
        l1_BW = model_l1_BW[category]
        l2_W  = model_l2_W[category]
        l2_BW = model_l2_BW[category]
        l1_U = layer(U,l1_W,l1_BW)
        l2_U = layer(l1_U,l2_W,l2_BW)
        out[category] = l2_U[0]
        # --------------------
        out[category] = l2_U[0]
    predicted = "-"
    prediction = (out > treshold) * 1
    for i in range(0,4):
        if prediction[i] == 1:
            predicted = str(np.argmax(out))
    
    return str(predicted),np.round(out,decimals=6)
# ----------------------------------------
