import pickle,gzip
import numpy as np
import torch
import os
from PIL import Image
from torch.autograd import Variable
import network
output_dir='/home/siplab/'
# with gzip.open('/home/siplab/50_images_data/traffic_file.pkl.gz', 'rb') as f:
#     train_set = pickle.load(f)

from unet2 import UNet
net=UNet(3)
# print len(train_set[1])
# start_step=0
# params = list(net.parameters())
# optimizer = torch.optim.SGD(params[:], lr=0.000001, momentum=0.9)
# end_step=880
trained_model = '/home/siplab/Unet_150.h5'
network.load_net(trained_model, net)
net.cuda()
net.eval()
# list=[]

im_test=Image.open('/home/siplab/test_images/2097.png').convert('LA')
im_test=np.array(im_test,dtype=float)
im_test=im_test[:,:,0]
print im_test
exit
im_test = np.expand_dims(im_test, axis=0)
im_test = np.expand_dims(im_test, axis=0)
im_test = torch.from_numpy(im_test).cuda()
im_test=Variable(im_test)
print im_test

x,x_drr=net(im_test.float())
drr1,drr2,drr3=torch.split(x_drr,1,1)
drr1=drr1.data.cpu().numpy()
drr1=drr1[0,0,:,:]
drr2=drr2.data.cpu().numpy()
drr2=drr2[0,0,:,:]
drr3=drr3.data.cpu().numpy()
drr3=drr3[0,0,:,:]
import scipy.misc
scipy.misc.imsave('/home/siplab/test_images/outfile1.jpg', drr1)
scipy.misc.imsave('/home/siplab/test_images/outfile2.jpg', drr2)
scipy.misc.imsave('/home/siplab/test_images/outfile3.jpg', drr3)
