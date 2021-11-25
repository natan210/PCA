import matplotlib
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    x = np.load(filename)
    a = x - np.mean(x, axis=0)
    return a
    # TODO: add your code here


def get_covariance(dataset):
    x = np.dot(np.transpose(dataset), dataset)
    y = 1 / (len(dataset) - 1)
    x = x * y
    return x
    # TODO: add your code here


def get_eig(S, m):
    n = len(S)
    x, y = eigh(S, eigvals=[n - m, n - 1])
    xs = x.argsort()
    sortedx = x[xs[::-1]]
    ret = [[0 for i in range(m)] for j in range(m)]
    i = 0
    while i < m:
        ret[i][i] = sortedx[i]
        i += 1
    y = np.fliplr(y)
    return ret, y
    # TODO: add your code here


def get_eig_perc(S, perc):
    n = len(S)
    a, b = get_eig(S, n)
    sum = np.trace(a)
    m = 0
    while m < n:
        if a[m][m] / sum > perc:
            m += 1
        else:
            break
    x, y = eigh(S, eigvals=[n - m, n - 1])
    xs = x.argsort()
    sortedx = x[xs[::-1]]
    m = len(x)
    ret = [[0 for i in range(m)] for j in range(m)]
    i = 0
    while i < m:
        ret[i][i] = sortedx[i]
        i += 1
    y = np.fliplr(y)
    return ret, y

    # TODO: add your code here


def project_image(img, U):
    x = np.dot(U, np.dot(img, U))
    return x
    # TODO: add your code here


def display_image(orig, proj):
    rorig = np.reshape(orig, (-1, 32))
    rproj = np.reshape(proj, (-1, 32))
    rorig = rorig.transpose()
    rproj = rproj.transpose()
    figure, axs = matplotlib.pyplot.subplots(nrows=1, ncols=2)
    axs[0].set_title('Original')
    axs[1].set_title('Projection')
    pos1 = axs[0].imshow(rorig, aspect='equal')
    figure.colorbar(pos1, ax=axs[0])
    pos2 = axs[1].imshow(rproj, aspect='equal')
    figure.colorbar(pos2, ax=axs[1])

    matplotlib.pyplot.show()

    # TODO: add your code here
