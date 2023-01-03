from func import Module

a = [7, -4, 5, 8, -10, 4, 5, 0]
b = [1, -1, 1, 1, -1, 1, 1, -1]

def test():
    for i in range(len(a)):
        assert Module.activate_sign(a[i]) is b[i]