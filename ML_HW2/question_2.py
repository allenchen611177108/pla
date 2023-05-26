import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold


def linear_regression(x, y):
    x = np.concatenate((np.ones((x.shape[0], 1)), x[:, np.newaxis]), axis=1)
    y = y[:, np.newaxis]
    W = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)

    return W


def MSE(ground_truth, prediction):
    result = np.square(ground_truth - prediction)
    result = result.sum(axis=0) / prediction.shape[0]

    return result


def training_error(x, y, W, degree):
    # 計算預測 y 值
    y_pred = np.ones((y.shape[0]))
    for i in range(0, y_pred.shape[0]):
        temp_y = 0
        for j in range(0, W.shape[0]):
            temp_y += W[j] * x[i]**j
        y_pred[i] = temp_y

    # 計算 MSE
    training_error = MSE(y, y_pred)

    # 印出 training error
    print("Degree = %d, Training error:%f" % (degree, training_error))


def cross_validation(num_folds, x, y, degree, ax):
    # 初始化 cross-validation method
    kf = KFold(n_splits=num_folds)

    # 儲存 5 folds cross error
    errors = []

    times = 1
    # 跑 5 folds cross validation
    for train_index, test_index in kf.split(x):
        # 將資料切分成 training data 和 test data
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        W = linear_regression(x_train, y_train)

        # 計算預測 y 值
        y_pred = np.ones((y_test.shape[0]))
        for i in range(0, y_pred.shape[0]):
            temp_y = 0
            for j in range(0, W.shape[0]):
                temp_y += W[j] * x_test[i]**j
            y_pred[i] = temp_y

        # 計算 MSE
        test_error = MSE(y_test, y_pred)

        # 將此輪 test error 儲存
        errors.append(test_error)

        # 畫出每一輪
        xs = np.linspace(-3, 3, 200)
        ys = W[0] + W[1] * xs
        # 繪製線
        text = "linear regression-" + str(times) + "time"
        ax.plot(xs, ys, label=text)
        times += 1

    # 印出每一輪的 test error
    for i in range(0, len(errors)):
        print("第 %d 輪的error:%f" % (i + 1, errors[i]))

    # 印出平均 test error
    print("Degree = %d, 5-fold cross_validation Average error:%f" %
          (degree, np.mean(errors)))
    print("--------------------------------------------------------")

    return 0


    #  main
    # ----------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------
    #  y = mx
m = 2  # parameter m

#  設定samples數量
samples = 15

#  根據直線, 產生加了noise的樣本, x落在[-3, 3]
x = np.linspace(-3, 3, 15)
y = m * x + np.random.normal(0, 1, samples)

# 製作 figure1
fig1 = plt.figure()

# 在 figure 上製作一個 axes
ax_1 = fig1.add_subplot(1, 1, 1)

ax_1.scatter(x, y)
# 繪製線, 產生亂數的直線
text = "produce line"
ax_1.plot([-3, 3], [-6, 6], color="black", label=text)
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20, rotation=0)

# -----------------------------------------------------------------------------------
# linear regression 執行
W_best_Linear = linear_regression(x, y)
xs = np.linspace(-3, 3, 200)
ys = W_best_Linear[0] + W_best_Linear[1] * xs

# 繪製線
text = "linear regression"
ax_1.plot(xs, ys, 'r', label=text)

# show training error & 5-fold cross validation
training_error(x, y, W_best_Linear, 1)
cross_validation(5, x, y, 1, ax_1)

# -------------------------------------------------------------------------------------
# # 顯示圖形
plt.legend(loc='upper left', fontsize=16)
plt.show()
