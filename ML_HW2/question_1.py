import matplotlib.pyplot as plt
import numpy as np


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
ax_1.plot([-3, 3], [-6, 6], color="black")  # 產生亂數的直線
plt.xlabel('x', fontsize=20)
plt.ylabel('y', fontsize=20, rotation=0)

plt.show()
