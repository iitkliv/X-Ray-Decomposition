import pickle,gzip
import numpy as np
# image1="/home/siplab/Downloads/test1.png"
# file = open("data.pkl", "wb")
# pickle.dump(image1, file)
# pickle.dump(image1, file)
# file.close()
#
# # Read from file.
# file = open("data.pkl", "rb")
# image1 = pickle.load(file)
# image2 = pickle.load(file)
# file.close()
#
# cache_file = os.path.join(cache_path, name + '_gt_roidb.pkl')
PIK = "/home/siplab/50_images_data/traffic_file.pkl"
# with open(PIK, "wb") as f:
#     pickle.dump(len(data), f)
#     for value in data:
#         pickle.dump(value, f)
# data2 = []
# with open(PIK, "rb") as f:
#     for _ in range(pickle.load(f)):
#         data2.append(pickle.load(f))
# print data2
with gzip.open('/home/siplab/50_images_data/traffic_file.pkl.gz', 'rb') as f:
    train_set = pickle.load(f)
# data = cPickle.load(PIK)
# print data
img=train_set[14]
img=img.reshape((150,450))
npa = np.asarray(img, dtype=np.float32)

import scipy.misc
scipy.misc.imsave('/home/siplab/Desktop/outfile.jpg', npa)
# plt.imshow(train_set[0].reshape((450, 150)), cmap=cm.Greys_r)
# plt.show()
print len(train_set)