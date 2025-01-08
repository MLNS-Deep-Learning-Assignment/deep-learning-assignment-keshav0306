import numpy as np
import matplotlib.pyplot as plt

root = "/ssd_scratch//cvit/keshav/DL-Project/"
train_data1 = np.load(root+'data2.npy')
train_lab1 = np.load(root+'lab2.npy')
print(train_data1.shape)
print(len(train_data1), len(train_lab1))
i = 3
plt.imshow(train_data1[i])
plt.savefig('img.png')
print(train_lab1[i])