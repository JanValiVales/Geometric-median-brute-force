import itertools
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Calculating geometric median from a random list of points."""
    # Generating random x and y coordinates of points (CHANGE HERE)
    x_intv = (1000, 5000)
    y_intv = (5000, 10000)
    n_points = 19

    x_arr = np.random.randint(low=x_intv[0], high=x_intv[1], size=n_points, dtype=np.int)
    y_arr = np.random.randint(low=y_intv[0], high=y_intv[1], size=n_points, dtype=np.int)

    # Setting initial constants and variables
    n_divisions = 10  # set int between 4 and 10
    lim_dist = 1E-4
    dif_dist = np.inf
    cur_dist = np.inf

    # Initial borders of given rectangular area
    minX, maxX = np.amin(x_arr), np.amax(x_arr)
    minY, maxY = np.amin(y_arr), np.amax(y_arr)

    # Graph init
    init_plot()


    i = 0
    while dif_dist >= lim_dist:
        xx = np.linspace(minX, maxX, n_divisions)
        yy = np.linspace(minY, maxY, n_divisions)

        all_points = list(itertools.product(xx, yy))

        x_size = (maxX - minX) / (n_divisions - 1)
        y_size = (maxY - minY) / (n_divisions - 1)

        i += 1
        min_dist = np.inf
        for point in all_points:
            total_dist = 0
            for given_point in zip(x_arr, y_arr):
                total_dist += l2_norm(np.asarray(given_point, dtype=np.float32), np.asarray(point, dtype=np.float32))

            if total_dist < min_dist:
                min_dist = total_dist
                minX, maxX = point[0] - x_size, point[0] + x_size
                minY, maxY = point[1] - y_size, point[1] + y_size

                dif_dist = abs(total_dist - cur_dist)
                cur_dist = total_dist
                gm = point
                output = "Iteration: %d, Distance: %f, Coordinates of geometric median: [%f, %f]" % (i, round(total_dist, 6), gm[0], gm[1])
                #Plot
                plot_gm(x_arr, y_arr, gm, total_dist)

    print(output)
    input("Press <Enter> to exit")


def init_plot():
    plt.xlim(1000)
    plt.ylim(1000)
    plt.ion()


def l2_norm(v, u):
    """Numpy distance between two points"""
    return np.linalg.norm(v - u)


def plot_gm(x_arr, y_arr, gm, total_dist):
    plt.clf()
    plt.scatter(x_arr, y_arr, c='blue', marker='o')
    plt.scatter(gm[0], gm[1], c='red', marker='*')

    for given_point in zip(x_arr, y_arr):
      plt.plot([given_point[0], gm[0]], [given_point[1], gm[1]], linewidth=0.5, c='black')

    plt.xticks([])
    plt.yticks([])
    plt.suptitle('Distance = %f' % total_dist, fontsize=20)
    #plt.show()
    #plt.savefig('body.png')
    plt.pause(0.01)


if __name__ == "__main__":
    main()