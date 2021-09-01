import tensorflow as tf
import numpy as np
from tensorflow.python.eager.backprop import GradientTape

d = np.array([0.00002, 0.00001])
L = np.array([1.326, 1.316])

x500: np.array = np.array(
    [[-4.2, 4.3],
    [-8.4, 8.5],
    [-12.1, 12.3]]
)


x1000: np.array = np.array(
    [[-8.5, 8.5],
    [-17.1, 17.0],
    [-25.8, 25.8]]
)

x500 = x500 * (10 ** (-2))
x1000 = x1000 * (10 ** (-2))

print(x500.shape)
delta4 = 0
for i in range(3):
    delta4 += x500[2-i][1] - x500[i][0]
#sum of 3 sections

delta4 = delta4 / 3
delta = delta4 / 4
print(delta)


x500 = np.mean(np.abs(x500), axis=1)
print(x1000)
x1000 = np.mean(np.abs(x1000), axis=1)

print(x1000)

sin = x1000 / np.sqrt(x1000*x1000 + L[1]*L[1])
print(sin)


def gradient(x500, x1000, d=d, L=L):

    #tmp
    #x1000 = x1000[0]


    x500 = tf.convert_to_tensor(x500)
    x1000 = tf.convert_to_tensor(x1000)
    #d = tf.convert_to_tensor(d[1])
    d = tf.convert_to_tensor(d[0])
    #L = tf.convert_to_tensor(L[1])

    #L = np.full(shape=(3,), fill_value=L[1])
    L = np.full(shape=(3,), fill_value=L[0])
    L = tf.convert_to_tensor(L)
    print(f"L:{L}")

    x1000 = x500


    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

        tape1.watch(x500)
        tape1.watch(x1000)
        tape1.watch(L)
        tape1.watch(d)

        print(f"x1000:{x1000}")
        sin = x1000 / tf.sqrt(x1000*x1000 + L*L)
        print(f"sin{sin}")
        m = tf.constant(np.array([1, 2, 3], dtype=np.float64))

        #tmp
        #m = tf.constant(np.array([1], dtype=np.float64))

        print(f"m:{m}, d;{d}")
        lambdas = d * (sin / m)
        #lambdas = lambdas * (10 ** 9)
        print(f"lambdas{lambdas}")

    #dsin_dx = tape.gradient(sin, x1000)
    #print(dsin_dx)
    #print(L*L / (tf.sqrt(x1000*x1000 + L*L) ** 3))
    dlambdas_dL = tape1.gradient(lambdas, L)
    print(f"dlamdas / dL \n{dlambdas_dL}")
    return

    dlambdas_dx = tape.gradient(lambdas, x1000)
    print(f"dlamdas / dx \n{dlambdas_dx}")
    return
    dlambdas_dL = tape.gradient(lambdas, L)
    print(f"dlamdas / dL \n{dlambdas_dL}")


def gradients(x500: np.ndarray, x1000: np.ndarray, d=d, L=L):
    print("=======gradients========")

    #tmp
    #x1000 = x1000[0]

    x500 = x500.reshape(1, 3)
    x1000 = x1000.reshape(1, 3)

    print(f"x500:{x500}")
    print(f"x1000:{x1000}")

    x = np.concatenate([x500, x1000])
    print(f"x:{x}")

    x = tf.convert_to_tensor(x)

    d = d.reshape((2, 1))
    d = tf.convert_to_tensor(d)

    #L = tf.convert_to_tensor(L[1])

    #L = np.full(shape=(3,), fill_value=L[1])
    L: np.ndarray = np.full(shape=(3,2), fill_value=L)
    L = L.transpose()
    L = tf.convert_to_tensor(L)
    print(f"L:{L}")

    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:

        for val in (x, d, L):
            tape1.watch(val)
            tape2.watch(val)

        print((x*x).shape)
        print((L*L).shape)
        sin = x / tf.sqrt(x*x + L*L)

        print(f"sin{sin}")

        m = tf.constant(np.array([1, 2, 3], dtype=np.float64))

        #tmp
        #m = tf.constant(np.array([1], dtype=np.float64))

        print(f"m:{m}, d;{d}")
        print((sin / m).shape)
        print(d.shape)
        lambdas = d * (sin / m)
        #lambdas = lambdas * (10 ** 9)
        print(f"lambdas{lambdas}")

    #dsin_dx = tape.gradient(sin, x1000)
    #print(dsin_dx)
    #print(L*L / (tf.sqrt(x1000*x1000 + L*L) ** 3))

    print("==========result===========")

    dlambdas_dL = tape1.gradient(lambdas, L)
    print(f"\ndlamdas / dL \n{dlambdas_dL}")

    dlambdas_dx = tape2.gradient(lambdas, x)
    print(f"\ndlamdas / dx \n{dlambdas_dx}")

    delta_by_x = dlambdas_dx * (0.05 * 10**(-2))
    delta_by_L = dlambdas_dL * 0.0005

    print()

    print(f"\nxの誤差:\n{delta_by_x}")
    print(f"\nLの誤差:\n{delta_by_L}")

    print(f"\nx_mean\n{tf.reduce_mean(delta_by_x, axis=1)}")
    print(f"\nL_mean\n{tf.reduce_mean(delta_by_L, axis=1)}")


    print()

    delta = delta_by_x + tf.abs(delta_by_L)
    print(f"\n誤差総和:\n{delta}")

    print(f"\n誤差平均:\n{tf.reduce_mean(delta, axis=1)}")



def main():
    gradients(x500, x1000)
    return

def get_lambda(d, x, L):
    sin = x / np.sqrt(x*x + L*L)

    m = np.array([1, 2, 3])

    lambdas = d * (sin / m)

    return lambdas


def abs():
    delta = 0.05
    lam_plus = get_lambda(d[0], x500 + delta, L[0])
    lam_minus = get_lambda(d[0], x500 - delta, L[0])

    print(f"delta_lambda:{np.abs(lam_plus - lam_minus) / 2}")



def test(L=L):
    print("=====test========")
    print(f"x1000{x1000}")
    print(f"square{x1000*x1000}")
    L = L[1]
    print(f"L:{L}")

    print()
    sin = x1000 / np.sqrt(x1000*x1000 + L*L)
    print(np.sqrt(x1000*x1000 + L*L))
    print(f"sin{sin}")


if __name__ == "__main__":
    main()
    #test()
    #abs()