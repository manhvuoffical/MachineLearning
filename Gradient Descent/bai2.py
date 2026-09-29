import numpy as np

# Bài 2: g(x) = (1/3)x^3 - x


def grad(x):
    return x ** 2 - 1


def cost(x):
    return (1 / 3) * x ** 3 - x


def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:  # just a small number
            break
        x.append(x_new)
    return (x, it)


(x, it) = myGD1(2, 0.1)  # x0 phai > -1
print("x = %.4f, g(x) = %.4f, so vong lap = %d" % (x[-1], cost(x[-1]), it))
