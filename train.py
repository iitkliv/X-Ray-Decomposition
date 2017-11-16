import pickle,gzip
import numpy as np
import torch
import os
from torch.autograd import Variable
import network
output_dir='/home/siplab/Desktop/'
with gzip.open('/home/siplab/50_images_data/traffic_file.pkl.gz', 'rb') as f:
    train_set = pickle.load(f)

from unet2 import UNet
net=UNet(3)
print len(train_set[1])
start_step=0
lr=0.001
params = list(net.parameters())
# optimizer = torch.optim.SGD(params[:], lr=lr, momentum=0.9)
optimizer=torch.optim.Adam(params, lr=lr, eps=1e-08)
end_step=880
net.cuda()
net.train()

# def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
#                       max_iter=100, power=0.9):
#     """Polynomial decay of learning rate
#         :param init_lr is base learning rate
#         :param iter is a current iteration
#         :param lr_decay_iter how frequently decay occurs, default is 1
#         :param max_iter is number of maximum iterations
#         :param power is a polymomial power
#
#     """
#     if iter % lr_decay_iter or iter > max_iter:
#         return optimizer
#
#     lr = init_lr*(1 - iter/max_iter)**power
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#     return lr

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)
list=[]
print net
for i in range(0,180):
    for step in range(start_step, end_step+1):
        count=step
        img=train_set[4*count]
        drr1=train_set[4*count+1]
        drr2=train_set[4*count+2]
        drr3=train_set[4*count+3]

        img = img.reshape((150, 450))
        drr1 = drr1.reshape((150, 450))
        drr2 = drr2.reshape((150, 450))
        drr3 = drr3.reshape((150, 450))
        img = np.asarray(img, dtype=np.float32)
        drr1 = np.asarray(drr1, dtype=np.float32)
        drr2 = np.asarray(drr2, dtype=np.float32)
        drr3 = np.asarray(drr3, dtype=np.float32)

        drr1 = np.expand_dims(drr1, axis=0)
        drr1 = np.expand_dims(drr1, axis=0)

        drr2 = np.expand_dims(drr2, axis=0)
        drr2 = np.expand_dims(drr2, axis=0)

        drr3 = np.expand_dims(drr3, axis=0)
        drr3 = np.expand_dims(drr3, axis=0)

        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)
        drr1 = torch.from_numpy(drr1)
        drr1 = Variable(drr1).cuda()


        drr2 = torch.from_numpy(drr2)
        drr2 = Variable(drr2).cuda()


        drr3 = torch.from_numpy(drr3)
        drr3=Variable(drr3).cuda()


        img = torch.from_numpy(img)
        img=Variable(img).cuda()

        net(img, drr1, drr2, drr3)
        loss = net.loss
        print loss
        optimizer.zero_grad()
        a=loss
        list.append(a.data.cpu().numpy()[0])
        loss.backward()
        # network.clip_gradient(net, 10.)
        optimizer.step()
    if i%30==0:
        save_name = os.path.join(output_dir, 'Unet_{}.h5'.format(i))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))

    scheduler.step()


thefile = open('/home/siplab/Desktop/loss_logger.txt', 'w')
for item in list:
  thefile.write("%s\n" % item)