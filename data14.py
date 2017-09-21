import numpy
import glob
import numpy as np
import cv2
import scipy.io as so
from  batch_augment import *

img_dir_list = glob.glob('../dataset2014/dataset/*/*/input/in000001.jpg')
#mask_dir_list = glob.glob('../dataset2014/dataset/*/*/groundtruth/gt000001.png')
img_dir_list = ['../dataset2014/dataset/'+img_dir.split('/')[3]+'/'+img_dir.split('/')[4]+'/' for img_dir in img_dir_list]

num_dirs = len(img_dir_list)
imgs_list = [glob.glob(img_dir+'input/*.jpg') for img_dir in img_dir_list]
masks_list = [glob.glob(img_dir+'groundtruth/*.png') for img_dir in img_dir_list]

num_imgs_list = [len(imgs) for imgs in imgs_list]
print num_imgs_list

print img_dir_list

def train_iter(batch_sz=12,img_sz=64,frames=5):
    while True:
        dir_idx = np.random.randint(0,num_dirs,batch_sz)
        idx_from = [np.random.randint(0, train_len-frames, 1)[0] for train_len in num_imgs_list]
        img_seqs = []
        mask_seqs = []

        for i in dir_idx:

            img_seq = [cv2.cvtColor(cv2.resize(cv2.imread(file),dsize=(img_sz,img_sz)),cv2.COLOR_RGB2GRAY)
                       for file in imgs_list[i][idx_from[i]:idx_from[i]+frames]]
            img_seqs.append(np.stack(img_seq,axis=-1))

            mask_seq = [cv2.blur((cv2.cvtColor(cv2.resize(cv2.imread(file),dsize=(img_sz,img_sz)),cv2.COLOR_RGB2GRAY)>178).astype(float),ksize=(3,3))
            for file in masks_list[i][idx_from[i]:idx_from[i]+frames]]
            mask_seqs.append(np.stack(mask_seq,axis=-1))

        batch = np.stack(img_seqs,axis=0)/127.5-1.0
        label = np.stack(mask_seqs)
        auged = get_auged_batch(np.concatenate([batch,label],axis=0))
        yield auged[:batch_sz],auged[-batch_sz:]
'''
for batch,label in train_iter():
    show = np.concatenate(batch[:,:,:,:3],axis=0)/2.0+0.5
    show_m = np.concatenate(label[:,:,:,:3],axis=0)/2.0+0.5
    print show.shape
    cv2.imshow('s',np.concatenate([show,show_m],axis=1))
    cv2.waitKey(300)
    print batch.shape,np.max(batch)#,np.min(batch)
'''