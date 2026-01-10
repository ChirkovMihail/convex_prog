import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path1 = 'L2error.txt'
    path2 = 'log_L2error.txt'
    f1 = open(path1, 'r')
    f2 = open(path2, 'r')
    
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []

    for line in f1:
        x, y = map(float, line.split())
        X1.append(x)
        Y1.append(y)
        
    fig, ax = plt.subplots()
        
    ax.plot(X1, Y1, label='L2 error')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('L2 error')
    ax.legend()
    ax.grid()
    plt.show()
        
    # for line in f2:
        # x, y = map(float, line.split())
        # X2.append(x)
        # Y2.append(y)
        
    # fig, ax = plt.subplots()
        
    # ax.plot(X2, Y2, label='log_L2 error')
    # ax.set_xlabel('Iterations')
    # ax.set_ylabel('log_L2 error')
    # ax.legend()
    # ax.grid()
    # plt.show()