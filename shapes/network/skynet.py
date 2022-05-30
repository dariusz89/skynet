import numpy as np
# ----------------------------------------
treshold = 0.5
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
l1_W  = np.loadtxt("network/net-data/net-model_trained-l1_W.csv", delimiter=',', ndmin=2)
l1_BW = np.loadtxt("network/net-data/net-model_trained-l1_BW.csv", delimiter=',', ndmin=1)
l2_W  = np.loadtxt("network/net-data/net-model_trained-l2_W.csv", delimiter=',', ndmin=2)
l2_BW = np.loadtxt("network/net-data/net-model_trained-l2_BW.csv", delimiter=',', ndmin=1)
# ----------------------------------------
def net(U):
	# Warstwa 1
    l1_S = np.dot(l1_W,U) + l1_BW
    l1_U = 1 / (1+np.exp(-l1_S))
    # --------------------
    # Warstwa 2
    l2_S = np.dot(l2_W,l1_U) + l2_BW
    exp_values = np.exp(l2_S - np.max(l2_S))
    l2_U = exp_values/np.sum(exp_values)
    # --------------------
    predicted = "-"
    prediction = (l2_U > treshold) * 1
    predicted = str(np.argmax(l2_U))
    return str(predicted),np.round(l2_U,decimals=6)
# ----------------------------------------
