import os
import math
import matplotlib.pyplot as plt

HOME_DIR = "/home/donaldo/PycharmProjects/ML/"
MESSAGES_DIR = HOME_DIR + "messages/"
START_OF_FILE = "Subject: "
START_LEN = len(START_OF_FILE)

ALPHAS = [10 ** x for x in range(-5, -2 - 2)]
N_GRAMS = [x for x in range(1 + 1, 4 - 1)]
LAMBDAS = [10 ** x for x in range(10)]
PARTS = [x for x in range(1, 11)]
PART_NAME = "part"

CLASSES = 2
EVENTS = 2


def split_data(leave_part, n):
    test_samples = []
    test_types = []
    train_samples = []
    train_types = []
    for part_ in os.listdir(MESSAGES_DIR):
        for file in os.listdir(MESSAGES_DIR + part_ + "/"):
            reader = open(MESSAGES_DIR + part_ + "/" + file)
            title = reader.readline()[START_LEN:].split()
            reader.readline()
            text = reader.readline().split()

            all_data = title.copy()
            for t in text:
                all_data.append(t)

            transformed_data = []
            for ii in range(len(all_data) - n + 1):
                transformed_data.append(tuple([all_data[ii + j] for j in range(n)]))

            data_type = 1 if "legit" in file else 0  # class = 0 -- spam, class = 1 -- legit

            if part_ == leave_part:
                test_samples.append(transformed_data)
                test_types.append(data_type)
            else:
                train_samples.append(transformed_data)
                train_types.append(data_type)

    return train_samples, train_types, test_samples, test_types


def predict(lambdas, intensity, sample_cs, sample_ws, tests):  # sample_c -- class, sample_w -- words, tests -- words
    aprior = [0.0] * CLASSES  # Pr(y), messages in y class
    likeli = {}  # p(x|y), messages with word x in y class
    # print(len(sample_cs))
    for ii in range(len(sample_cs)):
        aprior[sample_cs[ii]] += 1
        # print(sample_ws[:10])
        # print("words =", words)
        for word in sample_ws[ii]:
            if word not in likeli:
                likeli.update({word: [0.0] * CLASSES})
            likeli[word][sample_cs[ii]] += 1

    for ii in range(CLASSES):
        for word in likeli:
            likeli[word][ii] = math.log(likeli[word][ii] + intensity) - math.log(aprior[ii] + intensity * EVENTS)

    for ii in range(CLASSES):
        aprior[ii] /= len(sample_cs)

    # print("normalized")

    possibilities = []
    # print(len(tests))
    for test in tests:
        results_log = [0.0] * CLASSES  # p(class|X) = lambda * p(class) * p(X|class) [Mul p(word|class)]
        for y in range(CLASSES):
            if aprior[y] == 0:
                results_log[y] = -math.inf
                continue

            results_log[y] += math.log(lambdas[y] * aprior[y])
            for word in test:
                if word in likeli:
                    results_log[y] += likeli[word][y]

        max_result = max(results_log)
        sum_results = math.log(sum([math.exp(results_log[y] - max_result) for y in range(CLASSES)]))
        possibilities.append([math.exp(results_log[y] - max_result - sum_results) for y in range(CLASSES)])

    return possibilities


def drop(arr, new_len):
    res = []
    step = len(arr) // new_len
    for x in range(0, len(arr), step):
        res.append(arr[x])
    return res


best_accuracy, best_n, best_alpha, best_lambda = 0, 0, 0, 0
lambda_bound = 0
lambda_arr, accuracy_arr = [], []
best_FN = 1e10
for n_gram in N_GRAMS:
    for alpha in ALPHAS:
        curr_lambda_arr = []
        curr_accuracy_arr = []
        min_FN = 1e10
        for lam in LAMBDAS:
            print("N =", n_gram, "alpha =", alpha, "lambda power =", math.log10(lam))
            predictions = 0
            TP = 0
            TN = 0
            FN = 0
            for part in PARTS:
                # print("Calculating without part", part)
                train_s, train_t, test_s, test_t = split_data(PART_NAME + str(part), n_gram)

                how_legit = predict([1, lam], alpha, train_t, train_s, test_s)
                prediction = [0 if x[0] > x[1] else 1 for x in how_legit]

                predictions += len(prediction)
                for i in range(len(prediction)):
                    if prediction[i] == 1 and test_t[i] == 1:
                        TP += 1
                    elif prediction[i] == 0 and test_t[i] == 0:
                        TN += 1
                    elif prediction[i] == 0 and test_t[i] == 1:
                        FN += 1

            curr_accuracy = (TP + TN) / predictions

            if curr_accuracy > best_accuracy:
                print("Updated accuracy =", curr_accuracy, "lambda =", lam, "N =", n_gram, "alpha =", alpha)
                best_accuracy = curr_accuracy
                best_n = n_gram
                best_alpha = alpha
                best_lambda = lam

            curr_lambda_arr.append(lam)
            curr_accuracy_arr.append(curr_accuracy)
            if min_FN > FN:
                print("Found less FN =", FN, "at lambda power =", math.log10(lam))
                min_FN = FN
                lambda_bound = lam

        if best_FN > min_FN:
            best_FN = min_FN
            print("Updated FN =", min_FN, " when N =", n_gram, "alpha =", alpha, "lambda =", lam)
            print("lambda", curr_lambda_arr)
            print("accuracy", curr_accuracy_arr)
            second_err0_n = n_gram
            second_err0_alpha = alpha
            lambda_arr = curr_lambda_arr
            accuracy_arr = curr_accuracy_arr

legits, spams = 0, 0
best_f = []
for part in PARTS:
    train_s, train_t, test_s, test_t = split_data(PART_NAME + str(part), best_n)
    for i in test_t:
        if i == 0:
            spams += 1
        else:
            legits += 1
    curr_probability = predict([1, best_lambda], best_alpha, train_t, train_s, test_s)
    for p in range(len(curr_probability)):
        best_f.append([curr_probability[p], test_t[p]])

best_f.sort(key=lambda x: x[0][1], reverse=True)
FPR, TPR = [0], [0]
for i in range(len(best_f)):
    if best_f[i][1] == 0:
        FPR.append(FPR[i] + 1 / spams)
        TPR.append(TPR[i])
    else:
        TPR.append(TPR[i] + 1 / legits)
        FPR.append(FPR[i])

# plt.ylabel("TPR")
# plt.xlabel("FPR")
# plt.title("ROC, lambda = %f, alpha = %f, n_gram = %f" % (best_lambda, best_alpha, best_n))
# plt.plot(FPR, TPR)
# plt.show()

plt.ylabel("Accuracy")
plt.xlabel("Lambda")
plt.xscale('log')
plt.title("Accuracy-Lambda dependency")
plt.plot(lambda_arr, accuracy_arr)
plt.show()
