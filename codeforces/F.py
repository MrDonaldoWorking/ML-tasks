import math

classes = int(input())  # K
lambdas = list(map(int, input().split()))  # lambda
intensity = int(input())  # alpha
events = 2  # Q

sample_size = int(input())  # N
sample = []
aprior = [0] * classes  # Pr(y), messages in y class
likeli = {}  # p(x|y), messages with word x in y class
for i in range(sample_size):
    curr_input = input().split()
    curr_class = int(curr_input[0]) - 1

    aprior[curr_class] += 1
    for word in set(curr_input[2:]):
        if word not in likeli.keys():
            likeli.update({word: [0.0] * classes})
        likeli[word][curr_class] += 1

for i in range(classes):
    for word in likeli:
        likeli[word][i] = math.log(likeli[word][i] + intensity) - math.log(aprior[i] + intensity * events)

for i in range(classes):
    aprior[i] /= sample_size

test_size = int(input())
for i in range(test_size):
    curr_input = input().split()
    curr_words = set(curr_input[1:])

    # p(class|X) = lambda * p(class) * p(X|class) [Mul p(word|class)]
    results_log = [0.0] * classes
    for y in range(classes):
        if aprior[y] == 0:
            results_log[y] = -math.inf
            continue

        results_log[y] += math.log(lambdas[y] * aprior[y])
        for word in likeli.keys():
            if word in curr_words:
                results_log[y] += likeli[word][y]
            else:
                results_log[y] += math.log(1 - math.exp(likeli[word][y]))

    max_result = max(results_log)
    sum_results = math.log(sum([math.exp(results_log[y] - max_result) for y in range(classes)]))
    for y in range(classes):
        print('{:0.10f}'.format(math.exp(results_log[y] - max_result - sum_results)), end=' ')
    print()
