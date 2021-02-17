def div(dividend, divisor):
    if divisor == 0:
        return 0
    return dividend / divisor


k = int(input())

matrix = []
for i in range(k):
    matrix.append(list(map(int, input().split())))

All = 0
TP = []
FP = []
FN = []
classes = []
predicted = []
for i in range(k):
    TP.append(matrix[i][i])
    sum_FP = 0
    sum_FN = 0
    curr_class = 0
    sum_predicted = 0
    for j in range(k):
        All += matrix[i][j]
        curr_class += matrix[i][j]
        sum_predicted += matrix[j][i]
        if i != j:
            sum_FP += matrix[j][i]
            sum_FN += matrix[i][j]
    FP.append(sum_FP)
    FN.append(sum_FN)
    classes.append(curr_class)
    predicted.append(sum_predicted)
TN = []
for i in range(k):
    TN.append(All - TP[i] - FP[i] - FN[i])

# print("All", All)
# print("TP", TP)
# print("TN", TN)
# print("FP", FP)
# print("FN", FN)
#
# print("predicted", predicted)
# print("classes", classes)

Recall = []
Precision = []
for i in range(k):
    Recall.append(div(TP[i], (TP[i] + FN[i])))
    Precision.append(div(TP[i], (TP[i] + FP[i])))

# print("Recall", Recall)
# print("Precision", Precision)

F_score = []
for i in range(k):
    F_score.append(div(2 * Precision[i] * Recall[i], (Precision[i] + Recall[i])))

# print("F_score", F_score)

micro = 0
for i in range(k):
    micro += classes[i] * F_score[i]
micro = div(micro, All)

Precision_W = 0
Recall_W = 0
for i in range(k):
    Precision_W += div(TP[i] * classes[i], predicted[i])
    Recall_W += TP[i]
Precision_W = div(Precision_W, All)
Recall_W = div(Recall_W, All)

macro = div(2 * Precision_W * Recall_W, (Precision_W + Recall_W))

print('{:0.10f}'.format(macro))
print('{:0.10f}'.format(micro))
