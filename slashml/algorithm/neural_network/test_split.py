import numpy as np


""" a = np.arange(16).reshape((8, 2))
b = [2, 6, 7]
mask = np.ones(len(a), dtype=bool)
mask[b,] = False
x, y = a[b], a[mask] # instead of a[b] you could also do a[~mask]
print('\n', a)

indices = np.where(a[:, -1] < 7)


mask = np.random.choice([False, True], len(a), p=[0.75, 0.25])

#random_batch = np.random.choice(a.x.shape[0], a.x.shape[0] / 2)

print('\n', mask)

print(' mask data', a[mask]) """


#a = np.array([[1, 2], [3, 4]])
a = np.array([[1, 2]])
b = np.array([[5, 6]])
c = np.hstack((a, b))
d = c

print(c)