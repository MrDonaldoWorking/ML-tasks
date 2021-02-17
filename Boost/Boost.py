import pandas
import os
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import math
import numpy

HOME_DIR = "/home/donaldo/PycharmProjects/ML/"
BOOST_DIR = os.path.join(HOME_DIR, "Boost")

STEPS = [1, 2, 3, 5, 8, 13, 21, 34, 55]
MAX_STEP = max(STEPS)

PLOT_COLORS = "br"
PLOT_STEP = 0.02
CLASS_NAMES = "NP"
FIGURE_SIZE = 5
PLOT_SIZE = 100


def read_csv_data(file_name):
    data = pandas.read_csv(os.path.join(BOOST_DIR, file_name))
    return data[data.columns[:-1]].values, data[data.columns[-1]].values


def calc_accuracy(expected, predicted):
    cnt = 0
    for pos in range(len(expected)):
        if expected[pos] == predicted[pos]:
            cnt += 1
    return cnt / len(expected)


def ada_predict(ensemble, features):
    n, m = features.shape
    boosted_prediction = [0] * n
    for pair in ensemble:
        curr_prediction = pair[1].predict(features)
        for pos in range(n):
            boosted_prediction[pos] += pair[0] * curr_prediction[pos]
    return numpy.sign(boosted_prediction)


def do_boost(file_name):
    print(file_name)

    features, classes = read_csv_data(file_name)
    classes = [1 if curr_class == 'P' else -1 for curr_class in classes]

    x_min, x_max = features[:, 0].min(), features[:, 0].max()
    y_min, y_max = features[:, 1].min(), features[:, 1].max()
    x_step, y_step = (x_max - x_min) / PLOT_SIZE, (y_max - y_min) / PLOT_SIZE
    x_min -= x_step
    x_max += x_step
    y_min -= y_step
    y_max += y_step

    xs, ys = numpy.meshgrid(numpy.arange(x_min, x_max, x_step),
                            numpy.arange(y_min, y_max, y_step))

    weights = [1 / len(classes)] * len(classes)
    ensemble, accuracies = [], []
    for step in range(MAX_STEP):
        print("step", step + 1)
        curr_clf = DecisionTreeClassifier(max_depth=2).fit(features, classes, sample_weight=weights)
        prediction = curr_clf.predict(features)

        error = 0
        for pos in range(len(classes)):
            if prediction[pos] != classes[pos]:
                error += weights[pos]
        alpha = math.log((1 - error) / error) / 2
        # print("error", error)
        # print("alpha", alpha)

        for pos in range(len(classes)):
            weights[pos] *= math.exp(-alpha * prediction[pos] * classes[pos])
        w_sum = sum(weights)
        for pos in range(len(classes)):
            weights[pos] /= w_sum

        ensemble.append([alpha, curr_clf])

        boosted_prediction = ada_predict(ensemble, features)
        accuracies.append(calc_accuracy(classes, boosted_prediction))
        print(calc_accuracy(classes, boosted_prediction))

        if STEPS.count(step + 1) == 1:
            Z = ada_predict(ensemble, numpy.c_[xs.ravel(), ys.ravel()]).reshape(xs.shape)
            plt.figure(figsize=(FIGURE_SIZE, FIGURE_SIZE))
            cs = plt.contourf(xs, ys, Z, cmap=plt.cm.Paired)

            for i, n, c in zip([-1, 1], CLASS_NAMES, PLOT_COLORS):
                idx = numpy.where(numpy.array(classes) == i)
                plt.scatter(features[idx, 0], features[idx, 1],
                            c=c, cmap=plt.cm.Paired,
                            s=20, edgecolor='k',
                            label="Class %s" % n)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.legend(loc='upper right')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Decision Boundary, step %d, %s' % (step + 1, file_name))

            plt.tight_layout()
            plt.savefig("plots/boost/%s/step %d.png" % (file_name.replace('.csv', ''), step + 1))
            plt.show()

    plt.xlabel("AdaBoost step")
    plt.ylabel("Accuracy")
    plt.title("Step-Accuracy dependency %s" % file_name)
    plt.plot([x + 1 for x in range(MAX_STEP)], accuracies)
    plt.show()


for name in os.listdir(BOOST_DIR):
    do_boost(name)
