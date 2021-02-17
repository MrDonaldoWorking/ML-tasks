n, m, k = map(int, input().split())
array = list(map(int, input().split()))

classes = []
for i in range(m):
    classes.append([])
for i in range(n):
    classes[array[i] - 1].append(i)

result = []
for i in range(k):
    result.append([])

part_pos = 0
for i in range(m):
    for j in classes[i]:
        result[part_pos].append(j + 1)
        part_pos = (part_pos + 1) % k

for i in range(k):
    print(len(result[i]), *result[i])
