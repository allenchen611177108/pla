import random
import matplotlib.pyplot as plt

pos = []
neg = []
def positive(param_list):
    found = False
    while(found != True):
        y = float(random.randint(-500, 500))
        x = float(random.randint(-500, 500))
        x_inline = float((y-param_list[1])/param_list[0])
        if(x > x_inline):
            found = True
            break;
    return [x, y, 1]

def negative(param_list):
    found = False
    while(found != True):
        y = float(random.randint(-500, 500))
        x = float(random.randint(-500, 500))
        x_inline = float((y-param_list[1])/param_list[0])
        if(x < x_inline):
            found = True
            break;
    return [x, y, -1]

def false_positive(param_list):
    found = False
    while (found != True):
        y = float(random.randint(-500, 500))
        x = float(random.randint(-500, 500))
        x_inline = float((y - param_list[1]) / param_list[0])
        if (x < x_inline):
            found = True
            break;
    return [x, y, 1, -1]

def false_negative(param_list):
    found = False
    while (found != True):
        y = float(random.randint(-500, 500))
        x = float(random.randint(-500, 500))
        x_inline = float((y - param_list[1]) / param_list[0])
        if (x > x_inline):
            found = True
            break;
    return [x, y, -1, 1]

def generate(m, b, n):
    param = [float(m), float(b)]
    for i in range(n):
        pos.append(positive(param))
        neg.append(negative(param))

def graph():
    plt.xlabel('X')
    plt.ylabel('Y')
    x_p = []
    y_p = []
    x_n = []
    y_n = []
    for i in range(len(pos)):
        x_p.append(pos[i][0])
        y_p.append(pos[i][1])
    plt.scatter(x_p,y_p, c="red")
    for i in range(len(neg)):
        x_n.append(neg[i][0])
        y_n.append(neg[i][1])
    plt.scatter(x_n,y_n, c="blue")
    plt.show()
