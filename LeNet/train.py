from src.optimizer.SDLM import SDLM
from model import LeNet5
from src.utils.data_loader import load_data

def train(X, y):
    """
        Training based on LeNet-5 Architecture 
    """

    model = LeNet5()
    optimizer = SDLM()
    n_val = 5000

    #Set the learning rate based on layer as per the LeNet-5 paper
    lr_schedule = {
        'C1.W': 0.001,   # input → C1
        'C1.b': 0.002,   # bias = 2× weight LR
        'S2.alpha': 0.005,
        'S2.beta': 0.005,
        'C3.W0': 0.005, 'C3.W1': 0.005,'C3.W2': 0.005, 'C3.W3': 0.005,'C3.W4': 0.005,'C3.W5': 0.005,'C3.W6': 0.005,'C3.W7': 0.005,'C3.W8': 0.005,'C3.W9': 0.005,'C3.W10': 0.005,'C3.W11': 0.005,'C3.W12': 0.005,'C3.W13': 0.005,'C3.W14': 0.005,'C3.W15': 0.005,
        'C3.b': 0.01,
        'S4.alpha': 0.01,
        'S4.beta': 0.01,
        'C5.W': 0.01,
        'C5.b': 0.02,
        'F6.W': 0.01,
        'F6.b': 0.02,
    }
    for name, lr in lr_schedule.items():
        optimizer.set_lr_multiplier(name, lr)
    
    #Load the data
    X_train, X_test, y_train, y_test = load_data() #This will give images of shapes (batch, channel, height, width) = (batch, 1, 32, 32)

    #Splitting the training data to training and validation set
    X_val = X_train[:5000]
    y_val = y_train[:5000]

    X_train = X_train[5000: ]
    y_train = y_train[5000: ]

    #Training through epochs
    epoch = 20

    for i in range(epoch):


# if __name__ == "__main__":
#     train()