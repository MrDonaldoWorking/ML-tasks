import numpy
import pandas
# from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
# get_ipython().magic('matplotlib inline')
import math


def div(dividend, divisor):
    if divisor == 0:
        return 0
    return dividend / divisor


def dist(dist_type, param1, param2):
    res = 0
    if dist_type == "manhattan":
        for x in range(len(param1)):
            res += abs(param1[x] - param2[x])
    elif dist_type == "euclidean":
        for x in range(len(param1)):
            res += (param1[x] - param2[x]) ** 2
        res **= 1 / 2
    else:
        for x in range(len(param1)):
            res = max(res, abs(param1[x] - param2[x]))
    return res


def bounding(x, if_not_zero):
    if x < 1:
        return if_not_zero
    return 0


def kern(kernel_type, x):
    if kernel_type == "uniform":
        return bounding(x, 1 / 2)
    elif kernel_type == "triangular":
        return bounding(x, 1 - x)
    elif kernel_type == "epanechnikov":
        return bounding(x, 3 / 4 * (1 - x ** 2))
    elif kernel_type == "quartic":
        return bounding(x, 15 / 16 * (1 - x ** 2) ** 2)
    elif kernel_type == "triweight":
        return bounding(x, 35 / 32 * (1 - x ** 2) ** 3)
    elif kernel_type == "tricube":
        return bounding(x, 70 / 81 * (1 - x ** 3) ** 3)
    elif kernel_type == "gaussian":
        return math.exp(-(x ** 2) / 2) / ((2 * math.pi) ** (1 / 2))
    elif kernel_type == "cosine":
        return bounding(x, math.pi / 4 * math.cos(math.pi * x / 2))
    elif kernel_type == "logistic":
        return 1 / (math.exp(x) + 2 + math.exp(-x))
    else:
        return 2 / math.pi / (math.exp(x) + math.exp(-x))


def guess(dist_type, kern_type, window_type, h, params, vals, this_param):
    objects = []
    for i in range(len(params)):
        objects.append({
            "param": params[i],
            "val": vals[i]
        })
    objects.sort(key=lambda some_param: dist(dist_type, this_param, some_param["param"]))

    divisor = dist(dist_type, this_param, objects[h]["param"]) if window_type == "variable" else h
    this_val = numpy.zeros(len(vals[0]))
    kern_sum = 0
    for obj in objects:
        curr_w = 0
        curr_dist = dist(dist_type, this_param, obj["param"])
        if curr_dist == 0 or divisor != 0:
            curr_w = kern(kern_type, div(curr_dist, divisor))
        # print(obj, "w =", curr_w, "dist =", curr_dist, "window_val =", window_val)
        for i in range(len(vals[0])):
            this_val[i] += obj["val"][i] * curr_w
        kern_sum += curr_w

    for i in range(len(vals[0])):
        if kern_sum == 0:
            for obj in objects:
                this_val[i] += obj["val"][i]
            this_val[i] /= len(params)
        else:
            this_val[i] /= kern_sum

    return this_val


def run(dist_type, kern_type, window_type, h, params, vals):
    # Leave-one-out Cross-Validation
    # print(params)
    # print(params.columns)
    # print(params.iloc[0])
    # print(params.iloc[1].tolist()[0])
    k = len(vals.columns)
    matrix = [[0 for j in range(k)] for i in range(k)]
    for i in range(len(params)):
        curr_val = guess(dist_type, kern_type, window_type, h, params.drop(i, axis=0).values.tolist(),
                         vals.drop(i, axis=0).values.tolist(), params.iloc[i].tolist())
        matrix[int(numpy.argmax(vals.iloc[i]))][int(numpy.argmax(curr_val))] += 1

    All = 0
    TP = []
    classes = []
    predicted = []
    for i in range(k):
        TP.append(matrix[i][i])
        curr_class = 0
        sum_predicted = 0
        for j in range(k):
            All += matrix[i][j]
            curr_class += matrix[i][j]
            sum_predicted += matrix[j][i]
        classes.append(curr_class)
        predicted.append(sum_predicted)

    Precision_W = 0
    Recall_W = 0
    for i in range(k):
        Precision_W += div(TP[i] * classes[i], predicted[i])
        Recall_W += TP[i]
    Precision_W = div(Precision_W, All)
    Recall_W = div(Recall_W, All)

    return div(2 * Precision_W * Recall_W, (Precision_W + Recall_W))


def greatest_dist(dist_type, params):
    res = 0
    for i in range(len(params)):
        for j in range(len(params)):
            res = max(res, dist(dist_type, params[i], params[j]))

    return res


metrics = ['manhattan', 'euclidean', 'chebyshev']
kernels = ['uniform', 'triangular', 'epanechnikov', 'quartic']
windows = ['fixed', 'variable']
data = pandas.read_csv('data/dataset_54_vehicle.csv')
y_names = data['Class'].unique()

# Normalization
for clazz in data:
    # TODO: find all number types
    if data[clazz].dtype.name == 'int64':
        data[clazz] = (data[clazz] - min(data[clazz])) / (max(data[clazz]) - min(data[clazz]))

# OneHot
for clazz in data:
    if data[clazz].dtype.name == 'object':
        # print(data.drop(clazz, axis=1))
        # print(pandas.get_dummies(data[clazz]))
        data = pandas.concat([data.drop(clazz, axis=1), pandas.get_dummies(data[clazz])], axis=1)

data.to_csv('data/part-1.csv')

xs = data.drop(y_names, axis=1)
ys = data[y_names]

# print(params[:3])
# print(vals[:3])

best_metric = ''
best_kernel = ''
best_window = ''
best_window_size = 0
best_f_score = 0
for metric in metrics:
    max_dist = greatest_dist(metric, xs.values.tolist())
    for kernel in kernels:
        for window in windows:
            print(metric, kernel, window)
            if window == "variable":
                for h in range(1, math.ceil(len(xs) ** (1 / 2))):
                    print("running h=", h, "...")
                    curr_f_score = run(metric, kernel, window, h, xs, ys)
                    if curr_f_score > best_f_score:
                        best_metric = metric
                        best_kernel = kernel
                        best_window = window
                        best_window_size = h
                        best_f_score = curr_f_score
            else:
                curr_dist = max_dist / (len(xs) ** (1 / 2))
                while curr_dist <= max_dist:
                    print("running dist=", curr_dist, "...")
                    curr_f_score = run(metric, kernel, window, curr_dist, xs, ys)
                    if curr_f_score > best_f_score:
                        best_metric = metric
                        best_kernel = kernel
                        best_window = window
                        best_window_size = curr_dist
                        best_f_score = curr_f_score
                    curr_dist += max_dist / (len(xs) ** (1 / 2))
            # print(best_metric, best_kernel, best_window, best_window_size)
            # print("The best F-score is", best_f_score)

print(best_metric, best_kernel, best_window, best_window_size)
print("The best F-score is", best_f_score)

# For graphic
scores = []
window_sizes = []
max_dist = greatest_dist(best_metric, xs.values.tolist())
if best_window == "fixed":
    curr_dist = max_dist / (len(xs) ** (1 / 2))
    while curr_dist <= max_dist:
        scores.append(run(best_metric, best_kernel, best_window, curr_dist, xs, ys))
        window_sizes.append(curr_dist)
        curr_dist += max_dist / (len(xs) ** (1 / 2))
else:
    for h in range(1, math.ceil(len(xs) ** (1 / 2))):
        scores.append(run(best_metric, best_kernel, best_window, h, xs, ys))
        window_sizes.append(h)

plt.plot(window_sizes, scores)
plt.xlabel("window sizes")
plt.ylabel("F-score")
plt.show()
