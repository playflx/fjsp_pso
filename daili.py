import copy
import random
import numpy as np

print(random.random())
# # list1 = [8, 4, 5, 6, 7, 1, 3, 2]
# # list2 = [8, 7, 1, 2, 3, 5, 4, 6]
# # list1 = [5,2,6,3,1,7,4,7]
# # list2 = [4,5,0,4,0,1,3,5]
# list1 = [4., 5., 5., 1., 8., 2., 6., 0., 5., 6., 3., 7., 2., 9., 8., 4., 3., 5., 1., 2., 3., 8., 1., 8.,
#   6., 4., 7., 9., 7., 8., 0., 9., 7., 6., 4., 0., 3., 5., 1., 7., 2., 0., 3., 9., 8., 4. ,1. ,4.,
#   9., 0., 6., 9., 0. ,2. ,5.]
# p1=list(map(int,list1))
# list2 = [5., 3., 3. ,5. ,1., 7. ,9. ,0. ,5. ,3. ,8. ,9., 9. ,0. ,9. ,1., 8. ,1. ,0. ,8. ,3. ,2. ,5. ,4.,
#   6. ,6. ,7. ,6. ,8. ,9. ,4. ,4. ,7. ,6. ,5. ,2. ,2. ,2. ,7. ,4. ,5. ,8. ,6. ,1. ,0. ,9. ,7. ,8.,
#   2. ,4. ,3. ,4. ,0. ,0. ,1.]
# p2=list(map(int,list2))
# print("初始染色体p1:", p1)
# print("初始染色体p2:", p2)
# list1 = copy.deepcopy(p1)
# list2 = copy.deepcopy(p2)
# seq = [i+1 for i in range(10)]
# random_length1 = np.random.randint(2, len(seq)-1)
#
# for i in range(random_length1):
#     index = np.random.randint(0, len(seq))
#     seq.pop(index)
# set2 = set(seq)
# print("需要交换的工件工序",set2)
# # child1 = copy.deepcopy(p1)
# # child2 = copy.deepcopy(p2)
# # remain1 = [i for i in range(len(p1)) if p1[i] in set2]
# # remain2 = [i for i in range(len(p1)) if p2[i] in set2]
# # cursor1, cursor2 = 0, 0
# # for i in range(len(p1)):
# #     if list2[i] in set2:
# #         child1[remain1[cursor1]] = list2[i]
# #         cursor1 += 1
# #     if list1[i] in set2:
# #         child2[remain2[cursor2]] = list1[i]
# #         cursor2 += 1
# # for i in range(10):
# #     print('第一个子代的第{}个工件的数量:'.format(i),child1.count(i),p1.count(i))
# #
# # print()
# # for i in range(10):
# #     print('第二个子代的第{}个工件的数量:'.format(i), child2.count(i),p1.count(i))
#
# # flag = True
# #
# # while flag:
# #     k1 = random.randint(0,len(list1)-1)
# #     k2 = random.randint(0,len(list1)-1)
# #     if k1 < k2:
# #         flag = False
# # k1 = 21
# # k2 = 37
# # fragment1 = p1[k1:k2]
# # fragment2 = p2[k1:k2]
# # list1 = copy.deepcopy(p1)
# # list2 = copy.deepcopy(p2)
# # list1[k1:k2]=fragment2
# # list2[k1:k2]=fragment1
# # print("k1:",k1)
# # print("k2:",k2)
# # print("交换后的父本1",list1)
# # print("交换后的父本2",list2)
# # l1 = copy.deepcopy(list1)
# # del l1[k1:k2]
# # l2 = copy.deepcopy(list2)
# # del l2[k1:k2]
# # fra1 = copy.deepcopy(fragment1)
# # fra2 = copy.deepcopy(fragment2)
# # pos_index1 = [0]*10
# # pos_index2 = [0]*10
# # off1 =[]
# # for i in fragment1:
# #     pos_index1[i]+=1
# # for i in fragment2:
# #     pos_index2[i]+=1
# # ind = [0]*10
# # for i in range(10):
# #     ind[i] = pos_index1[i]-pos_index2[i]  # 判断截断后的每一个工序少了或多了几个
# # for pos in l1:
# #     if pos in fra2:
# #         while ind[pos] < 0:  # <0 表示该工序变多了，需要去掉
# #             temp = fra1[fra2.index(pos)]  # 找到fra1中对应的元素
# #             if ind[temp] > 0:  # 大于零说明截断后的工序集合少了该工序，可以替换掉工序变多的pos
# #                 off1.append(temp)
# #                 ind[temp]-=1
# #                 ind[pos]+=1  # 替换后，减少了的工序相应+1，加了的工序相应-1
# #             elif ind[temp] < 0:  #  小于零说明截断后的工序集合多该工序，不可以替换掉工序变多的pos，去掉
# #     else:
# #         off1.append(pos)
# #
# # offspring1 = []
# # pos_index = [0]*10
# # for i in fragment2:
# #     pos_index[i] +=1
# # for pos in l1:
# #     # pos_index[pos] += 1
# #     if pos in fragment2:
# #         if pos_index[pos]<p1.count(pos):
# #             offspring1.append(pos)
# #             pos_index[pos] += 1
# #             continue
# #         pos_1 = fragment1[fragment2.index(pos)]
# #         while pos_1 not in fragment2 or pos_index[pos_1] == p1.count(pos_1) :
# #             if pos_1==pos:
# #                 if pos_1==0:
# #                     pos_1 = fragment1[fragment2.index(pos_1 + 1)]
# #                 else:
# #                     pos_1 = fragment1[fragment2.index(pos_1 - 1)]
# #             else:
# #                 pos_1 = fragment1[fragment2.index(pos_1)]
# #
# #             # else:
# #             #     a=1
# #             #     break
# #         offspring1.append(pos_1)
# #         pos_index[pos_1] += 1
# #         continue
# #
# #     offspring1.append(pos)
# #     pos_index[pos] += 1
# # k11 = k1
# # for i in range(len(fragment2)):
# #     offspring1.insert(k11,fragment2[i])
# #     k11+=1
# # print('交叉后的子代1:',offspring1)
# # for i in range(10):
# #     print('第{}个工件的数量:'.format(i),offspring1.count(i))
# #     print(pos_index[i])
# #     print(p1.count(i))
# #
# # offspring2 = []
# # pos_index = [0]*10
# # for i in fragment1:
# #     pos_index[i] +=1
# # for pos in l2:
# #     if pos in fragment1:
# #         pos_index[pos]+=1
# #         if pos_index[pos]<=list2.count(pos):
# #             offspring2.append(pos)
# #             continue
# #         pos_1 = fragment2[fragment1.index(pos)]
# #         while pos_1 in fragment1 and pos_1==pos:
# #             pos_1 = fragment2[fragment1.index(pos_1)]
# #         offspring2.append(pos)
# #         continue
# #     offspring2.append(pos)
# # k21 = k1
# # for i in range(len(fragment1)):
# #     offspring2.insert(k21,fragment1[i])
# #     k21+=1
# # print('交叉后的子代2:',offspring2)
# # for i in range(10):
# #     print('第{}个工件的数量:'.format(i),offspring2.count(i))
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # #
# # # fragment1_only=[y for y in fragment1 if y not in fragment2]
# # # fragment2_only=[z for z in fragment2 if z not in fragment1]
# # # for pos1 in fragment2_only:
# # #     l1[l1.index(pos1)] = fragment1_only[fragment2_only.index(pos1)]
# # # for pos2 in fragment1_only:
# # #     l2[l2.index(pos2)] = fragment2_only[fragment1_only.index(pos2)]