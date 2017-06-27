import matplotlib.pyplot as plt
import numpy as np

g = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1],
              [0, 0, 0, 0, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 1, 1, 1, 0],
              [0, 0, 0, 0, 0, 1, 1, 0, 1]])

fig, ax = plt.subplots(1,1)

ax.imshow(g, cmap='Greys',  interpolation='nearest')
x = ['1', '4', '6', '5', '8' ,'2', '3', '7', '9']
plt.xticks(range(len(x)), x, fontsize=30)

y = ['1', '4', '6', '5', '8' ,'2', '3', '7', '9']
plt.yticks(range(len(y)), y, fontsize=30)

plt.show()