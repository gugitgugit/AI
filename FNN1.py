import numpy as np

def setup_network():
    net = {}
    net['W1'] = np.array([[0.3, 0.1, 0.23], [0.21, 0.5, 0.1]])
    net['b1'] = np.array([0.2, 0.1, 0.5])

    net['W2'] = np.array([[0.1, 0.4, 0.2], [0.4, 0.2, 0.4], [0.5, 0.4, 0.3]])
    net['b2'] = np.array([0.3, 0.1, 0.4])

    net['W3'] = np.array([[0.1, 0.23], [0.5, 0.2], [0.1, 0.13]])
    net['b3'] = np.array([0.2, 0.1])

    return net
                         
def predict(net, input) :
    W1, W2, W3 = net['W1'], net['W2'], net['W3']
    b1, b2, b3 = net['b1'], net['b2'], net['b3']

    v1 = np.dot(input, W1) + b1
    h1 = sigmoid(v1)
    print("v1:\n", v1), print("h1:\n", h1)
    
    v2 = np.dot(h1, W2) + b2
    h2 = sigmoid(v2)
    print("v2:\n", v2), print("h2:\n", h2)
    
    v3 = np.dot(h2, W3) + b3
    h3 = sigmoid(v3)
    print("v3:\n", v3), print("h3:\n", h3)

    y = identify(h3)
    return y

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def identify(x) :
    return x

FNN = setup_network()

x = np.array([[2.2, 1.3],
              [1.2, 2.3]])

print("Input : \n", x)

y = predict(FNN, x)

print("Output : \n", y)