from dataset.svw import SVW
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_CLASSES = [
    SVW.ARCHERY,
    SVW.BASEBALL,
    SVW.BASKETBALL,
    SVW.BMX,
    SVW.BOWLING,
    SVW.BOXING,
    SVW.CHEERLEADING,
    SVW.DISCUSTHROW,
    SVW.DIVING,
    SVW.FOOTBALL,
    SVW.GOLF,
    SVW.GYMNASTICS,
    SVW.HAMMERTHROW,
    SVW.HIGHJUMP,
    SVW.HOCKEY,
    SVW.HURDLING,
    SVW.JAVELIN,
    SVW.LONGJUMP,
    SVW.POLEVAULT,
    SVW.ROWING,
    SVW.RUNNING,
    SVW.SHOTPUT,
    SVW.SKATING,
    SVW.SKIING,
    SVW.SOCCER,
    SVW.TENNIS,
    SVW.VOLLEYBALL,
    SVW.WEIGHT,
    SVW.WRESTLING,
    SVW.SWIMMING,
]


def print_shapes(x_train, y_train, x_test, y_test):
    for inp in x_train:
        print(inp.shape)
        print(np.array(y_train.shape))
        print('=' * 30)
        for inp in x_test:
            print(inp.shape)
        print(np.array(y_test.shape))
        print('=' * 30)


def plot_histogram(histogram):
    plt.xticks(rotation=90)
    histogram_train = dict(sorted(histogram.items()))
    plt.bar(histogram_train.keys(), [p[0] for p in histogram_train.values()])
    plt.bar(histogram_train.keys(), [p[1] for p in histogram_train.values()])
