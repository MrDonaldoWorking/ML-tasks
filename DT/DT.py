import os
import pandas
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import random

HOME_DIR = "/home/donaldo/PycharmProjects/ML/"
DT_DIR = os.path.join(HOME_DIR, "DT")

CRITERIONS = ['gini', 'entropy']
SPLITTERS = ['best', 'random']
DEPTHS = [x for x in range(1, 30)]

DATA_NUMBERS = [x for x in range(1, len(os.listdir(DT_DIR)) // 2)]
TEST_NAME = "test"
TRAIN_NAME = "train"


def get_file_name(num, file_type):
    return os.path.join(DT_DIR, num + "_" + file_type + ".csv")


def get_csv_data(file_name):
    data = pandas.read_csv(file_name)
    return data[data.columns[:-1]].values, data[data.columns[-1]].values


def get_all_csv_data(num):
    return get_csv_data(get_file_name(num, TRAIN_NAME)), get_csv_data(get_file_name(num, TEST_NAME))


def draw_accuracy_height_graph(hyper_params):
    num, _depth, _criterion, _splitter = hyper_params
    (train_X, train_Y), (test_X, test_Y) = get_all_csv_data(num)

    train_accuracies, test_accuracies = [], []
    for h in DEPTHS:
        tree = DecisionTreeClassifier(criterion=_criterion, splitter=_splitter, max_depth=h).fit(train_X, train_Y)
        train_accuracies.append(tree.score(train_X, train_Y))
        test_accuracies.append(tree.score(test_X, test_Y))

    fig, ax = plt.subplots()
    plt.xlabel("Height")
    plt.ylabel("Accuracy")
    plt.title("Accuracy-Height dependency, " + _criterion + ", " + _splitter + ", " + num + " dataset")
    ax.plot(DEPTHS, train_accuracies, 'g', label="train")
    ax.plot(DEPTHS, test_accuracies, 'r', label="test")
    ax.legend(loc=4, shadow=True, fontsize='large')
    plt.show()


def bootstrap(train_X, train_Y):
    res_X, res_Y = [], []
    for n in range(len(train_Y)):
        curr_rand = random.randint(0, len(train_Y) - 1)
        res_X.append(train_X[curr_rand])
        res_Y.append(train_Y[curr_rand])
    return res_X, res_Y


def find_most_frequent(arr):
    res = None
    max_qty = 0
    for x in arr:
        curr_qty = arr.count(x)
        if curr_qty > max_qty:
            max_qty = curr_qty
            res = x
    return res


def vote(arr_of_arr):
    res = []
    for pos in range(len(arr_of_arr[0])):
        all_votes = [arr[pos] for arr in arr_of_arr]
        res.append(find_most_frequent(all_votes))
    return res


def calc_accuracy(expected, predictions):
    predict = vote(predictions)
    cnt = 0
    for pos in range(len(expected)):
        if expected[pos] == predict[pos]:
            cnt += 1
    return cnt / len(expected)


results = []
for i in DATA_NUMBERS:
    curr_num_str = str(i).zfill(2)
    (train_features, train_classes), (test_features, test_classes) = get_all_csv_data(curr_num_str)

    best_accuracy, best_depth = -1, 0
    best_criterion, best_splitter = None, None
    for criterion in CRITERIONS:
        for splitter in SPLITTERS:
            print(i, criterion, splitter)
            for depth in DEPTHS:
                curr_tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth).fit(train_features, train_classes)
                accuracy = curr_tree.score(test_features, test_classes)

                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    best_depth = depth  # curr_tree.get_depth()
                    best_criterion = criterion
                    best_splitter = splitter
    results.append([curr_num_str, best_depth, best_criterion, best_splitter])

results.sort(key=lambda x: x[1])
print(results[0][0], "dataset with minimal height =", results[0][1])
draw_accuracy_height_graph(results[0])
print(results[-1][0], "dataset with maximal height =", results[-1][1])
draw_accuracy_height_graph(results[-1])

for i in DATA_NUMBERS:
    curr_num_str = str(i).zfill(2)
    print(curr_num_str, "dataset")
    (train_features, train_classes), (test_features, test_classes) = get_all_csv_data(curr_num_str)

    features_in_use = int(len(train_features[0]) ** (1 / 2))

    for criterion in CRITERIONS:
        for splitter in SPLITTERS:
            train_predicts, test_predicts = [], []
            for tree in range(len(train_features[0])):
                curr_features, curr_classes = bootstrap(train_features, train_classes)
                curr_tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_features=features_in_use)
                curr_tree = curr_tree.fit(curr_features, curr_classes)

                train_predicts.append(curr_tree.predict(train_features))
                test_predicts.append(curr_tree.predict(test_features))

            train_accuracy = calc_accuracy(train_classes, train_predicts)
            test_accuracy = calc_accuracy(test_classes, test_predicts)
            print("Forest accuracy with " + criterion + ", " + splitter + ": train =", train_accuracy, "test =", test_accuracy)
