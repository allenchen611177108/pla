import random
import time
from sklearn import datasets
from func import Module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Generate_sample import generate, pos, neg


data_num = 2000
generate(3, 2, data_num)
data = pos +neg
random.shuffle(data)

def pla():
    w = np.array([0., 0., 0.])
    error = 1
    iterator = 0
    while error != 0:
        error = 0
        for i in range(len(data)):
            input = np.concatenate((np.array([1.]), np.array(data[i])[:2]))
            target = float(data[i][2])

            if Module.activate_sign(np.dot(w, input)) != target:
                error = 1
                w += target*input
        iterator += 1
    print(w)

def pocket():
    w_p = np.array([0., 0., 0.])
    error_num = data_num
    while error_num != 0:
        w = w_p
        index = random.randint(0, data_num)
        rand_input = np.concatenate((np.array([1.]), np.array(data[index])[:2]))
        rand_target = float(data[index][2])
        w += rand_target*rand_input
        error = 0
        for i in range(len(data)):
            input = np.concatenate((np.array([1.]), np.array(data[index])[:2]))
            target = float(data[i][2])
            if Module.activate_sign(np.dot(w, input)) != target:
                error += 1
        if(error < error_num):
            error_num = error
            w_p = w
    print(w_p)

start = time.time()
pocket()
end = time.time()
print(format(end-start))