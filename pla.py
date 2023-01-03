from sklearn import datasets
from func import Module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
y = pd.DataFrame(data=iris['target'], columns=['target'])
iris_data = pd.concat([x,y], axis=1) # Combine DataFrame objects horizontally along the x axis by passing in axis=1
iris_data['target'] = iris_data['target'].map({0:'setosa',
                                               1:'versicolor',
                                               2:'virginica'
                                               })
iris_data = iris_data[(iris_data['target'] == 'setosa') | (iris_data['target'] == 'versicolor')]
iris_data = iris_data[['sepal length (cm)', 'petal length (cm)', 'target']]

iris_data['target'] = iris_data['target'].map({'setosa':1,
                                               'versicolor':-1})

w = np.array([0., 0., 0.])
error = 1
iterator = 0
while error != 0:
    error = 0
    for i in range(len(iris_data)):

        plt.xlabel('sepal_length')
        plt.ylabel('petal_length')
        plt.plot(iris_data['sepal length (cm)'], iris_data['petal length (cm)'], 'mo')

        print("iterator" + str(iterator))
        iterator += 1
        error += 1

        x = np.concatenate((np.array([1.]), np.array(iris_data.iloc[i])[:2]))
        y = np.array(iris_data.iloc[i])[2]

        if w[1] != 0:
            x_last_decision_boundary = np.linspace(0,w[1])
            y_last_decision_boundary = (w[2]/w[1])*x_last_decision_boundary
            plt.plot(x_last_decision_boundary, y_last_decision_boundary, 'c--')

        if Module.activate_sign(np.dot(w, x)) != y:
            w += y*x
        print("y: " + str(y))
        print("x: " + str(x))
        print("w: " + str(w))

        x_vector = np.linspace(0,x[1])
        y_vector = (x[2]/x[1])*x_vector
        plt.plot(x_vector, y_vector, 'b')

        x_decision_boundary = np.linspace(-0.5, 7)
        y_decision_boundary = (-w[1]/w[2])*x_decision_boundary - (w[0]/w[2])
        plt.plot(x_decision_boundary, y_decision_boundary, 'r')

        plt.show()
