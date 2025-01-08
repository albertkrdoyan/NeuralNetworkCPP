import matplotlib.pyplot as plt
plt.plot([float(x) for x in open('plot.txt').readlines()])
plt.show()