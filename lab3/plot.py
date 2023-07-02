import numpy as np
import matplotlib.pyplot as plt 

def plot_acc(Relu_train_acc, Relu_test_acc, LeakyRelu_train_acc, LeakyRelu_test_acc, ELU_train_acc, ELU_test_acc):
    epochs = [i for i in range (500)]
    plt.figure()
    plt.plot(epochs, Relu_train_acc, label='Relu_train_acc')
    plt.plot(epochs, Relu_test_acc, label='Relu_test_acc')
    plt.plot(epochs, LeakyRelu_train_acc, label='LeakyRelu_train_acc')
    plt.plot(epochs, LeakyRelu_test_acc, label='LeakyRelu_test_acc')
    plt.plot(epochs, ELU_train_acc, label='ELU_train_acc')
    plt.plot(epochs, ELU_test_acc, label='ELU_test_acc')
    plt.legend(loc='upper right')
    plt.ylabel('Accuracy')  
    plt.legend()    
    plt.savefig('Result.png')
