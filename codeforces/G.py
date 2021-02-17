import math
from collections import deque

vertices_cnt = 0


class Node:
    def __init__(self, class_type, left, right, feature_index, predicate, is_list):
        self.class_type = class_type
        self.left = left
        self.right = right
        self.feature_index = feature_index
        self.predicate = predicate
        self.is_list = is_list


def id3(curr_objects, depth):
    global vertices_cnt

    cnt = {}
    for obj in curr_objects:
        if obj["class_t"] not in cnt:
            cnt[obj["class_t"]] = 0
        cnt[obj["class_t"]] += 1

    if len(cnt) == 1:
        vertices_cnt += 1
        return Node(curr_objects[0]["class_t"], None, None, None, None, True)

    best_entropy, best_pred, best_feature = -1, 0, 0
    for feature in range(len(curr_objects[0]["feature"])):
        curr_objects.sort(key=lambda o: o["feature"][feature])
        curr_cnt = cnt.fromkeys(cnt.keys(), 0)
        for pos in range(len(curr_objects)):
            if pos > 0 and curr_objects[pos]["feature"][feature] == curr_objects[pos - 1]["feature"][feature]:
                curr_cnt[curr_objects[pos]["class_t"]] += 1
                continue

            left_entropy, right_entropy = 0, 0
            for p in range(pos):
                val = curr_cnt[curr_objects[p]["class_t"]] / pos
                left_entropy -= val * math.log2(val)
            for p in range(len(curr_objects) - pos):
                val = (cnt[curr_objects[p + pos]["class_t"]] - curr_cnt[curr_objects[p + pos]["class_t"]]) / (len(curr_objects) - pos)
                right_entropy -= val * math.log2(val)
            curr_entropy = (left_entropy * pos + right_entropy * (len(curr_objects) - pos)) / len(curr_objects)

            if best_entropy == -1 and best_pred == 0 or curr_entropy < best_entropy:
                best_entropy = curr_entropy
                best_pred = curr_objects[pos]["feature"][feature]
                best_feature = feature

            curr_cnt[curr_objects[pos]["class_t"]] += 1

    left_objects, right_objects = [], []
    for obj in curr_objects:
        if obj["feature"][best_feature] < best_pred:
            left_objects.append(obj)
        else:
            right_objects.append(obj)

    if len(left_objects) == 0 or len(right_objects) == 0:
        majority = curr_objects[0]["class_t"]
        max_value = cnt[majority]
        for obj in curr_objects:
            if cnt[obj["class_t"]] > max_value:
                max_value = cnt[obj["class_t"]]
                majority = obj["class_t"]

        vertices_cnt += 1
        return Node(majority, None, None, None, None, True)
    else:
        left_v = id3(left_objects, depth + 1)
        right_v = id3(right_objects, depth + 1)
        vertices_cnt += 1
        return Node(None, left_v, right_v, best_feature, best_pred, False)


def bfs(root_v):
    q = deque()
    q.append(root_v)
    num = 1
    while len(q) != 0:
        curr_v = q.popleft()
        if curr_v.is_list:
            print("C", curr_v.class_type)
        else:
            print("Q", curr_v.feature_index + 1, curr_v.predicate, 2 * num, 2 * num + 1)
            q.append(curr_v.left)
            q.append(curr_v.right)
        num += 1


features, classes, max_depth = map(int, input().split())
objects_qty = int(input())
objects = []
for i in range(objects_qty):
    inputs = list(map(int, input().split()))
    curr_class_type = inputs.pop()
    objects.append({
        "feature": inputs,
        "class_t": curr_class_type
    })

root = id3(objects, 0)

print(vertices_cnt)
bfs(root)
