import numpy as np

# Bài 1: f(x) = x^2 - 2


def grad(x):
    return 2 * x


def cost(x):
    return x ** 2 - 2


def myGD1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        if abs(grad(x_new)) < 1e-3:  # just a small number
            break
        x.append(x_new)
    return (x, it)


(x, it) = myGD1(5, 0.1)
print("x = %.4f, f(x) = %.4f, so vong lap = %d" % (x[-1], cost(x[-1]), it))
