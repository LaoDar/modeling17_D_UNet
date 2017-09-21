import numpy
import glob
import numpy as np
import cv2
import scipy.io as so
from batch_augment import *
train_dirs = ['birds','boats','bottle','zodiac','cyclists','flock','freeway','hockey']
test_dirs = ['jump','landing','ocean','peds','rain','skiing','surf','surfers']

train_paths = ['./JPEGS/'+dir+'/' for dir in train_dirs]
test_paths = ['./JPEGS/'+dir+'/' for dir in test_dirs]

def get_mask(dir,idx_from,idx_to,img_sz=64):
    m = so.loadmat('./GT_matlab/'+dir+'_GT.mat')
    m = m[m.keys()[0]]
    #print m.shape
    mask = np.stack([cv2.resize(m[:,:,i],dsize=(img_sz,img_sz)) for i in range(idx_from,idx_to)],axis=-1)
    return mask

def get_l(dir):
    m = so.loadmat('./GT_matlab/' + dir + '_GT.mat')
    m = m[m.keys()[0]]
    return m.shape[-1]

def get_file_list(path):
    return glob.glob(path+'*.jpg')

train_lst = [get_file_list(path) for path in train_paths]
test_lst = [get_file_list(path) for path in test_paths]
l_train = [get_l(dir) for dir in train_dirs]
l_test = [get_l(dir) for dir in test_dirs]
train_lens = [min(len(lst),l) for lst,l in zip(train_lst,l_train)]
test_lens = [min(len(lst),l) for lst,l in zip(test_lst,l_test)]
#print train_lens,test_lens

def train_iter(img_sz=64,frames=5):
    while True:
        idx_from = [np.random.randint(0, train_len-frames, 1)[0] for train_len in train_lens]
        #print train_lens,idx_from
        #print idx_from
        img_seqs = []
        mask_seqs = []
        for i in range(8):
            img_seq = [cv2.cvtColor(cv2.resize(cv2.imread(file),dsize=(img_sz,img_sz)),cv2.COLOR_RGB2GRAY)
                       for file in train_lst[i][idx_from[i]:idx_from[i]+frames]]
            img_seq = np.stack(img_seq,axis=-1)
            img_seqs.append(img_seq)
            #print idx_from[i]
            #print train_dirs[i]
            mask_seqs.append(get_mask(train_dirs[i],idx_from[i],idx_from[i]+frames))

        batch = np.stack(img_seqs,axis=0)/127.5-1.0
        label = np.stack(mask_seqs)
        auged = get_auged_batch(np.concatenate([batch, label], axis=0))
        yield auged[:8], auged[-8:]

def test_iter(img_sz=64,frames=5):
    while True:
        idx_from = [np.random.randint(0, test_len-frames, 1)[0] for test_len in test_lens]
        #print idx_from
        img_seqs = []
        mask_seqs = []
        for i in range(8):
            img_seq = [cv2.cvtColor(cv2.resize(cv2.imread(file),dsize=(img_sz,img_sz)),cv2.COLOR_RGB2GRAY)
                       for file in test_lst[i][idx_from[i]:idx_from[i]+frames]]
            img_seq = np.stack(img_seq,axis=-1)
            img_seqs.append(img_seq)
            mask_seqs.append(get_mask(test_dirs[i], idx_from[i], idx_from[i] + frames))
        batch = np.stack(img_seqs,axis=0)/127.5-1.0
        label = np.stack(mask_seqs)
        auged = get_auged_batch(np.concatenate([batch, label], axis=0))
        yield auged[:8], auged[-8:]

'''
for batch,label in test_iter():
    show = np.concatenate(batch[:,:,:,:3],axis=0)/2.0+0.5
    show_m = np.concatenate(label[:,:,:,:3],axis=0)/2.0+0.5
    #print show.shape
    cv2.imshow('s',np.concatenate([show,show_m],axis=1))
    cv2.waitKey(300)
    #print batch.shape,np.max(batch)#,np.min(batch)


'''