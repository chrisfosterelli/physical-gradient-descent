""" Execute gradient descent on earth geometry """

import csv
import rasterio
import argparse
import numpy as np
from pylab import plot, show, xlabel, ylabel

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='output.csv')
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--decay', type=float, default=0.99)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--iters', type=int, default=10000)
parser.add_argument('--lat', type=float, default=47.801686)
parser.add_argument('--lon', type=float, default=-123.709083)
parser.add_argument('--tif', type=str, default='tifs/srtm_12_03.tif')
args = parser.parse_args()

src = rasterio.open(args.tif)
band = src.read(1)


def get_elevation(lat, lon):
    vals = src.index(lon, lat)
    return band[vals]


def compute_cost(theta):
    lat, lon = theta[0], theta[1]
    j = get_elevation(lat, lon)

    return j


def calculate_gradient(theta, j_history, n_iter):
    try:
        cost = compute_cost(theta)

        elev1 = get_elevation(theta[0] + 0.001, theta[1])  # north
        elev2 = get_elevation(theta[0] - 0.001, theta[1])  # south
        elev3 = get_elevation(theta[0], theta[1] + 0.001)  # east
        elev4 = get_elevation(theta[0], theta[1] - 0.001)  # west
    except IndexError:
        print('The boundary of elevation map has been reached')
        return None

    j_history[n_iter] = [max(0, cost), theta[0], theta[1]]
    if cost <= 0:
        return None

    lat_slope = elev1 / elev2 - 1
    lon_slope = elev3 / elev4 - 1
    print('Elevation at', theta[0], theta[1], 'is', cost)

    return (lat_slope, lon_slope)


def gradient_descent(theta, alpha, num_iters):
    j_history = np.zeros(shape=(num_iters, 3))

    for i in range(num_iters):
        slope = calculate_gradient(theta, j_history, i)
        if not slope:
            break

        print(f'({i}/{num_iters}): Update is ({slope[0]}, {slope[1]})')
        theta[0][0] += - alpha * slope[0]
        theta[1][0] += - alpha * slope[1]

    return theta, j_history[:i]


def gradient_descent_w_momentum(theta, alpha, gamma, num_iters):
    j_history = np.zeros(shape=(num_iters, 3))
    velocity = [0, 0]

    for i in range(num_iters):
        slope = calculate_gradient(theta, j_history, i)
        if not slope:
            break

        velocity[0] = gamma * velocity[0] - alpha * slope[0]
        velocity[1] = gamma * velocity[1] - alpha * slope[1]

        print(f'({i}/{num_iters}): Update is ({velocity[0]}, {velocity[1]})')

        theta[0][0] += velocity[0]
        theta[1][0] += velocity[1]

    return theta, j_history[:i]


def gradient_descent_w_nesterov(theta, alpha, gamma, num_iters):
    j_history = np.zeros(shape=(num_iters, 3))
    velocity = [0, 0]
    v_prev = [0, 0]

    for i in range(num_iters):
        slope = calculate_gradient(theta, j_history, i)
        if not slope:
            break

        v_prev = velocity[:]

        velocity[0] = gamma * velocity[0] - alpha * slope[0]
        velocity[1] = gamma * velocity[1] - alpha * slope[1]

        theta[0][0] += -gamma * v_prev[0] + (1 + gamma) * velocity[0]
        theta[1][0] += -gamma * v_prev[1] + (1 + gamma) * velocity[1]

    return theta, j_history[:i]


def adagrad(theta, alpha, epsilon, num_iters):
    j_history = np.zeros(shape=(num_iters, 3))
    cache = [0, 0]

    for i in range(num_iters):
        slope = calculate_gradient(theta, j_history, i)
        if not slope:
            break

        cache[0] += slope[0]**2
        cache[1] += slope[1]**2

        theta[0][0] += -alpha * slope[0] / (np.sqrt(cache[0]) + epsilon)
        theta[1][0] += -alpha * slope[1] / (np.sqrt(cache[1]) + epsilon)

    return theta, j_history[:i]


def RMSprop(theta, alpha, epsilon, decay_rate, num_iters):
    j_history = np.zeros(shape=(num_iters, 3))
    cache = [0, 0]

    for i in range(num_iters):
        slope = calculate_gradient(theta, j_history, i)
        if not slope:
            break

        cache[0] = decay_rate * cache[0] + (1 - decay_rate) * slope[0]**2
        cache[1] = decay_rate * cache[1] + (1 - decay_rate) * slope[1]**2

        theta[0][0] += -alpha * slope[0] / (np.sqrt(cache[0]) + epsilon)
        theta[1][0] += -alpha * slope[1] / (np.sqrt(cache[1]) + epsilon)

    return theta, j_history[:i]


def adam(theta, alpha, epsilon, beta1, beta2, num_iters):
    j_history = np.zeros(shape=(num_iters, 3))
    m, v = [0, 0], [0, 0]
    mt, vt = [0, 0], [0, 0]  # bias correction

    for i in range(num_iters):
        slope = calculate_gradient(theta, j_history, i)
        if not slope:
            break

        t = i + 1

        m[0] = beta1 * m[0] + (1 - beta1) * slope[0]
        m[1] = beta1 * m[1] + (1 - beta1) * slope[1]
        mt[0] = m[0] / (1 - beta1**t)
        mt[1] = m[1] / (1 - beta1**t)

        v[0] = beta2 * v[0] + (1 - beta2) * slope[0]**2
        v[1] = beta2 * v[1] + (1 - beta2) * slope[1]**2
        vt[0] = v[0] / (1 - beta2**t)
        vt[1] = v[1] / (1 - beta2**t)

        theta[0][0] += -alpha * mt[0] / (np.sqrt(vt[0]) + epsilon)
        theta[1][0] += -alpha * mt[1] / (np.sqrt(vt[1]) + epsilon)

    return theta, j_history[:i]


def genetic_alg(theta):
    pass


theta = np.array([[args.lat], [args.lon]])

# theta, j_history = gradient_descent(theta, args.alpha, args.iters)
# theta, j_history = gradient_descent_w_momentum(theta, args.alpha,
#                                                args.gamma, args.iters)
# theta, j_history = gradient_descent_w_nesterov(theta, args.alpha,
#                                                args.gamma, args.iters)
# theta, j_history = adagrad(theta, args.alpha, args.eps, args.iters)
theta, j_history = RMSprop(theta, args.alpha, args.eps, args.decay, args.iters)
# theta, j_history = adam(theta, args.alpha, args.eps,
#                         args.beta1, args.beta2, args.iters)

with open(args.output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for weight in j_history:
        if weight[1] != 0 and weight[2] != 0:
            writer.writerow([weight[1], weight[2]])


plot(np.arange(j_history.shape[0]), j_history[:, 0])
xlabel('Iterations')
ylabel('Elevation')
show()
