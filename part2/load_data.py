import numpy as np
import matplotlib.pyplot as plt

root = "/ssd_scratch//cvit/keshav/DL-Project/"
train_data1 = np.load(root+'data2.npy')
train_lab1 = np.load(root+'lab2.npy')
for i in range(10000):
    # i = 4
    plt.imshow(train_data1[i])
    plt.savefig(f'/ssd_scratch/cvit/keshav/vis_imgs/{i}.png')
    print(train_lab1[i])