import pickle,gzip
import numpy as np
import torch
import os
from torch.autograd import Variable
import network
output_dir='/home/siplab/'
with gzip.open('/home/siplab/50_images_data/traffic_file.pkl.gz', 'rb') as f:
    train_set = pickle.load(f)

from unet2 import UNet
net=UNet(3)
print len(train_set[1])
start_step=0
params = list(net.parameters())
optimizer = torch.optim.SGD(params[:], lr=0.00001, momentum=0.9)
end_step=3
net.cuda()
net.train()
list=[]
print net
# for i in range(0,21):
for step in range(start_step, end_step+1):
    count=step
    img=train_set[4*count]
    drr1=train_set[4*count+1]
    drr2=train_set[4*count+2]
    drr3=train_set[4*count+3]

    img=img.reshape((150,450))
    drr1=drr1.reshape((150,450))
    drr2=drr2.reshape((150,450))
    drr3=drr3.reshape((150,450))
    img = np.asarray(img, dtype=np.float32)
    drr1 = np.asarray(drr1, dtype=np.float32)
    drr2 = np.asarray(drr2, dtype=np.float32)
    drr3 = np.asarray(drr3, dtype=np.float32)
    import scipy.misc

    import scipy.misc

    scipy.misc.imsave('/home/siplab/test_images/outfile1.jpg', drr1)
    scipy.misc.imsave('/home/siplab/test_images/outfile2.jpg', drr2)
    scipy.misc.imsave('/home/siplab/test_images/outfile3.jpg', drr3)


    # drr1 = np.expand_dims(drr1, axis=0)
    # drr1 = np.expand_dims(drr1, axis=0)
    #
    # drr2 = np.expand_dims(drr2, axis=0)
    # drr2 = np.expand_dims(drr2, axis=0)
    #
    # drr3 = np.expand_dims(drr3, axis=0)
    # drr3 = np.expand_dims(drr3, axis=0)
    #
    # img = np.expand_dims(img, axis=0)
    # img = np.expand_dims(img, axis=0)
    # drr1 = torch.from_numpy(drr1)
    # drr1 = Variable(drr1).cuda()
    #
    #
    # drr2 = torch.from_numpy(drr2)
    # drr2 = Variable(drr2).cuda()
    #
    #
    # drr3 = torch.from_numpy(drr3)
    # drr3=Variable(drr3).cuda()
    #
    #
    # img = torch.from_numpy(img)
    # img=Variable(img).cuda()
    #
    # net(img, drr1, drr2, drr3)
    # loss = net.loss
    # print loss
    # optimizer.zero_grad()
    # a=loss
    # list.append(a.data.cpu().numpy()[0])
    # loss.backward()
    # network.clip_gradient(net, 10.)
    # optimizer.step()

#     save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(i))
#     network.save_net(save_name, net)
#     print('save model: {}'.format(save_name))
#
#
# thefile = open('/home/siplab/Desktop/loss_logger.txt', 'w')
# for item in list:
#   thefile.write("%s\n" % item)