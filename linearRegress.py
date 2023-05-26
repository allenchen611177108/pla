import random
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

def data(num):
    Data_X = []
    Data_Y = []
    for i in range(num):
        theta = random.random()
        x_set = []
        x = random.uniform(-3,3)
        y = 2*x + theta
        x_set.append(x)
        # x_set.append(theta)
        Data_X.append(x)
        Data_Y.append(y)
    return Data_X, Data_Y

LR = LinearRegression()
kf = KFold(n_splits=5)

Data = data(15)
X = numpy.array(Data[0])
print(X)
Y = numpy.array(Data[1])
print(Y)
error = []

xfit = []
yfit = []
for train_index, test_index in kf.split(X):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    LR.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
    y_pred = LR.predict(x_test.reshape(-1, 1))

    yfit.append(y_pred)
    plt.scatter(X, Y)
    plt.plot(x_test, y_pred)
    plt.show()
    error.append(mean_squared_error(y_test, y_pred))

print(error)
