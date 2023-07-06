import random

import numpy as np
num = range(0,100)
length = len(num)
nums = random.sample(range(len(num)),10)
nums=np.array(nums)

print(num)
print(nums)
print(length)
# import copy
#
# import numpy as np
# child1 = copy.deepcopy(parent1)
# child2 = copy.deepcopy(parent2)
# a, b = np.random.choice(np.arange(1, length), 2, replace=False)
# min_a_b, max_a_b = min([a, b]), max([a, b])
# if r_a_b is None:
#         r_a_b = range(min_a_b, max_a_b)
# r_left = np.delete(range(length), r_a_b)
# left_1, left_2 = child1[r_left], child2[r_left]
# middle_1, middle_2 = child1[r_a_b], child2[r_a_b]
# child1[r_a_b], child2[r_a_b] = middle_2, middle_1
# mapping = [[], []]
# for i, j in zip(middle_1, middle_2):
#     if j in middle_1 and i not in middle_2:
#         index = np.argwhere(middle_1 == j)[0, 0]
#         value = middle_2[index]
#         while True:
#             if value in middle_1:
#                 index = np.argwhere(middle_1 == value)[0, 0]
#                 value = middle_2[index]
#             else:
#                 break
#         mapping[0].append(i)
#         mapping[1].append(value)
#     elif i in middle_2:
#         pass
#     else:
#         mapping[0].append(i)
#         mapping[1].append(j)
# for i, j in zip(mapping[0], mapping[1]):
#     if i in left_1:
#         left_1[np.argwhere(left_1 == i)[0, 0]] = j
#     elif i in left_2:
#         left_2[np.argwhere(left_2 == i)[0, 0]] = j
#     if j in left_1:
#         left_1[np.argwhere(left_1 == j)[0, 0]] = i
#     elif j in left_2:
#         left_2[np.argwhere(left_2 == j)[0, 0]] = i
# child1[r_left], child2[r_left] = left_1, left_2
# return r_a_b, mapping,child1, child2
