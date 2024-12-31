import matplotlib.pyplot as plt
import os

arr = []

reader = open('plot.txt', 'r')
arr = reader.readlines()
reader.close()
#os.remove('plot.txt')

for i in range(len(arr)):
    arr[i] = float(arr[i])

plt.plot(arr)
plt.show()
