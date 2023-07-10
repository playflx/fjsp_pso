import copy
list1 = [4., 5., 5., 1., 8., 2., 6., 0., 5., 6., 3., 7., 2., 9., 8., 4., 3., 5., 1., 2., 3., 8., 1., 8.,
  6., 4., 7., 9., 7., 8., 0., 9., 7., 6., 4., 0., 3., 5., 1., 7., 2., 0., 3., 9., 8., 4. ,1. ,4.,
  9., 0., 6., 9., 0. ,2. ,5.]
p1 = list(map(int,list1))
print('交叉前:',p1)
l1 = copy.deepcopy(p1)
fragment2_only=[2,5,5,2,2]
fragment1_only=[1,0,0,3,3]
a = copy.deepcopy(fragment1_only)
b = copy.deepcopy(fragment2_only)
for pos1 in fragment2_only:
   l1[l1.index(pos1)] = a[b.index(pos1)]
   del a[0]
   del b[0]
print('交叉后:',l1)

# for pos2 in fragment1_only:
#     print(fragment2_only[fragment1_only.index(pos2)])