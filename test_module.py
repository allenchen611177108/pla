from func import Module

a = [7, -4, 5, 8, -10, 4, 5, 0]
b = [1, -1, 1, 1, -1, 1, 1, -1]

def test_sign_positive():
    assert Module.activate_sign(7) is 1

def test_sign_negative():
    assert Module.activate_sign(-1) is -1

def test_sign_zero():
    assert Module.activate_sign(0) is -1