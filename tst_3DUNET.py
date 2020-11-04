#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:39:05 2020

@author: apramanik
"""


from torch.utils.data import DataLoader
import numpy as np
import os, torch
import matplotlib.pyplot as plt

from tst_dataset import cardiacdata
from UNET3D_D4 import UNet3D



def dice_comp(pred, gt):
    return (2. * (np.sum(pred.astype(float) * gt.astype(float))) + 1.) / (np.sum(pred.astype(float)) \
        + np.sum(gt.astype(float)) + 1.)



#%%
nImg=1
dispind=0
vol_slice=5
chunk_size=nImg
#%% Choose training model directory
############################## 3DUNET #########################
subDirectory='20Apr_1115am_70I_10000E_1B'

print(subDirectory)

#%%
cwd=os.getcwd()
PATH= cwd+'/savedModels/'+subDirectory #complete path

#%%
tst_dataset = cardiacdata()
tst_loader = DataLoader(tst_dataset, batch_size=1, shuffle=False, num_workers=0)
# network
net = UNet3D(num_classes=4, in_channels=1, depth=4, start_filts=32, res=True).cuda()
net.load_state_dict(torch.load(os.path.join(PATH, "model_best.pth.tar"))['state_dict'])
normOrg=np.zeros((1,16,144,144),dtype=np.float32)
normGT=np.zeros((1,16,144,144),dtype=np.int16)
normSeg=np.zeros((1,16,144,144),dtype=np.int16)
dice = np.zeros((nImg, 3))
net.eval()
for step, (img, seg_gt) in enumerate(tst_loader, 0):
    img, seg_gt = img.cuda(), seg_gt.cuda()
    pred = net(img)
    _, pred = torch.max(pred, 1)

    pred = pred.squeeze().detach().cpu().numpy().astype(np.int8)
    img = img.squeeze().detach().cpu().numpy()
    gt = seg_gt.squeeze().detach().cpu().numpy().astype(np.int8)
    for i in range(3):
            dice[step, i] = dice_comp(pred==i+1, gt==i+1)
    normOrg[step]=img
    normGT[step]=gt
    normSeg[step]=pred


    
print("DICE Right Ventricle: {0:.5f}".format(np.mean(dice[:,0])))
print("DICE Myocardium: {0:.5f}".format(np.mean(dice[:,1])))
print("DICE Left Ventricle: {0:.5f}".format(np.mean(dice[:,2])))



#%%%
normOrg=np.reshape(normOrg,[int(normOrg.shape[1]/8),8,144,144])
normGT=np.reshape(normGT,[int(normGT.shape[1]/8),8,144,144])
normSeg=np.reshape(normSeg,[int(normSeg.shape[1]/8),8,144,144])
normError=np.abs(normGT.astype(np.float32)-normSeg.astype(np.float32))
normOrg=normOrg-normOrg.min()
#%% Display the output images
plot= lambda x: plt.imshow(x,cmap=plt.cm.gray,interpolation='bilinear')
plot1= lambda x: plt.imshow(x,interpolation='bilinear')
plt.clf()
plt.subplot(141)
plot(np.abs(normOrg[dispind,vol_slice,:,:]))
plt.axis('off')
plt.title('Original')
plt.subplot(142)
plot1(np.abs(normGT[dispind,vol_slice,:,:]))
plt.axis('off')
plt.title('True labels')
plt.subplot(143)
plot1(np.abs(normSeg[dispind,vol_slice,:,:]))
plt.axis('off')
plt.title('Segmentation')
plt.subplot(144)
plot(np.abs(normError[dispind,vol_slice,:,:]))
plt.title('Error')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=.01)
plt.show()



























