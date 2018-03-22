
""" Execute gradient descent on earth geometry """

import sys
import csv
import rasterio
import argparse
import numpy as np
from pylab import plot, show, xlabel, ylabel

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='output.csv')
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--iters', type=int, default=1000)
parser.add_argument('lat', type=float)
parser.add_argument('lon', type=float)
parser.add_argument('tif', type=str)
args = parser.parse_args()

src = rasterio.open(args.tif)
band = src.read(1)

def get_elevation(lat, lon):
    vals = src.index(lon, lat)
    return band[vals]

def compute_cost(theta):
    lat, lon = theta[0], theta[1]
    J = get_elevation(lat, lon)
    return J

def gradient_descent(theta, alpha, gamma, num_iters):
    J_history = np.zeros(shape=(num_iters, 3))
    velocity = [ 0, 0 ]

    for i in range(num_iters):

        try:
            cost = compute_cost(theta)

            elev1 = get_elevation(theta[0] + 0.001, theta[1])
            elev2 = get_elevation(theta[0] - 0.001, theta[1])
            elev3 = get_elevation(theta[0], theta[1] + 0.001)
            elev4 = get_elevation(theta[0], theta[1] - 0.001)
        except IndexError:
            print('The boundary of elevation map has been reached')
            break

        J_history[i] = [ cost, theta[0], theta[1] ]
        if cost <= 0: return theta, J_history

        lat_slope = elev1 / elev2 - 1
        lon_slope = elev3 / elev4 - 1 

        velocity[0] = gamma * velocity[0] + alpha * lat_slope
        velocity[1] = gamma * velocity[1] + alpha * lon_slope
        
        print('Update is', velocity[0])
        print('Update is', velocity[1])
        print('Elevation at', theta[0], theta[1], 'is', cost)

        theta[0][0] = theta[0][0] - velocity[0]
        theta[1][0] = theta[1][0] - velocity[1]

    return theta, J_history

theta = np.array([ [args.lat], [args.lon] ])
theta, J_history = gradient_descent(theta, args.alpha, args.gamma, args.iters)

with open(args.output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for weight in J_history:
        if weight[1] != 0 and weight[2] != 0:
            writer.writerow([ weight[1], weight[2] ])

plot(np.arange(args.iters), J_history[:, 0])
xlabel('Iterations')
ylabel('Elevation')
show()
