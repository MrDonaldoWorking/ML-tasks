import math


def div(dividend, divisor):
    if divisor == 0:
        return 0
    return dividend / divisor


def dist(param1, param2):
    res = 0
    if dist_type == "manhattan":
        for x in range(m):
            res += abs(param1[x] - param2[x])
    elif dist_type == "euclidean":
        for x in range(m):
            res += (param1[x] - param2[x]) ** 2
        res **= 1 / 2
    else:
        for x in range(m):
            res = max(res, abs(param1[x] - param2[x]))
    return res


def bounding(x, if_not_zero):
    if x < 1:
        return if_not_zero
    return 0


def kern(x):
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


def cmp(o):
    return dist(this_param, o["param"])


n, m = map(int, input().split())

objects = []
for i in range(n):
    param = list(map(int, input().split()))
    val = param.pop()
    objects.append({
        "param": param,
        "val": val
    })

this_param = list(map(int, input().split()))

dist_type = input()
kernel_type = input()
window_type = input()

h = 0
k = 0
if window_type == "fixed":
    h = int(input())
else:
    k = int(input())

objects.sort(key=cmp)
k_plus_1th_dist = dist(this_param, objects[k]["param"])
# print("k+1th dist =", k_plus_1th_dist)

divisor = 0
if window_type == "fixed":
    divisor = h
else:
    divisor = k_plus_1th_dist
# for obj in objects:
#     print(obj, "dist = ", cmp(obj))

this_val = 0
kern_sum = 0
for obj in objects:
    curr_w = 0
    curr_dist = cmp(obj)
    if curr_dist == 0 or divisor != 0:
        curr_w = kern(div(curr_dist, divisor))
    # print(obj, "w =", curr_w, "dist =", curr_dist, "window_val =", window_val)
    this_val += obj["val"] * curr_w
    kern_sum += curr_w

if kern_sum == 0:
    for obj in objects:
        this_val += obj["val"]
    this_val /= n
else:
    this_val /= kern_sum

print('{:0.10f}'.format(this_val))
