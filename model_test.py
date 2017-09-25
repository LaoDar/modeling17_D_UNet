import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import numpy as np
import cv2
from ops import *
from data import *
from BGGenerator import *
import data14
import os

normal = tf.truncated_normal_initializer

def get_UNet_params():
    wc = []
    wc.append(tf.get_variable('wc0', [4, 4, 3, 64], initializer=normal(stddev=0.02)))  # 32
    wc.append(tf.get_variable('wc1', [4, 4, 64, 128], initializer=normal(stddev=0.02)))  # 16
    wc.append(tf.get_variable('wc2', [4, 4, 128, 256], initializer=normal(stddev=0.02)))  # 8
    wc.append(tf.get_variable('wc3', [4, 4, 256, 512], initializer=normal(stddev=0.02)))  # 4
    wc.append(tf.get_variable('wc4', [4, 4, 512, 512], initializer=normal(stddev=0.02)))  # 2
    #wc.append(tf.get_variable('wc5', [2, 2, 256, 256], initializer=normal(stddev=0.02)))  # 1
    wd = []
    #wd.append(tf.get_variable('wd0', [2, 2, 256, 256], initializer=normal(stddev=0.02)))
    wd.append(tf.get_variable('wd1', [4, 4, 512, 512], initializer=normal(stddev=0.02)))
    wd.append(tf.get_variable('wd2', [4, 4, 256, 512 * 2], initializer=normal(stddev=0.02)))
    wd.append(tf.get_variable('wd3', [4, 4, 128, 256 * 2], initializer=normal(stddev=0.02)))
    wd.append(tf.get_variable('wd4', [4, 4, 64, 128 * 2], initializer=normal(stddev=0.02)))  # 2
    wd.append(tf.get_variable('wd5', [4, 4, 1, 64], initializer=normal(stddev=0.02)))  # 2
    wd.append(tf.get_variable('wd6', [1], initializer=normal(stddev=0.02)))  # 2
    return wc, wd

def get_UNet(s,wd):
    w,d = wd
    enc0 = conv2x2(s, w[0])
    enc1 = batch_norm(conv2x2(relu(enc0), w[1]))
    enc2 = batch_norm(conv2x2(relu(enc1), w[2]))
    enc3 = batch_norm(conv2x2(relu(enc2), w[3]))
    enc4 = batch_norm(conv2x2(relu(enc3), w[4]))
    #enc5 = batch_norm(conv2x2(lrelu(enc4), w[5]))
    #enc5 = tf.concat([enc5],3)
    #dec0 = batch_norm(deconv2x2(lrelu(enc5), d[0]))
    #dec0 = tf.concat([dec0, enc4],3)
    dec1 = batch_norm(deconv2x2(lrelu(enc4), d[0]))
    dec1 = tf.concat([dec1, enc3],3)
    dec2 = batch_norm(deconv2x2(lrelu(dec1), d[1]))
    dec2 = tf.concat([dec2, enc2],3)
    dec3 = batch_norm(deconv2x2(lrelu(dec2), d[2]))
    dec3 = tf.concat([dec3,enc1],3)
    dec4 = batch_norm(deconv2x2(lrelu(dec3), d[3]))
    future = 1.1*tf.nn.sigmoid(deconv2x2(lrelu(dec4), d[4])+d[5])-0.05   #it will be sigmoid!
    return future

def video_iter(path,batch_sz=8,img_sz=64,frames=3):
    cap = cv2.VideoCapture(path)
    ret,frame = cap.read()
    frame = cv2.cvtColor(cv2.resize(frame,dsize=(img_sz,img_sz)),cv2.COLOR_BGR2GRAY)
    batch_frames = [frame]
    while ret:
        if len(batch_frames) < batch_sz+frames-1:
            ret, frame = cap.read()
            try:frame = cv2.cvtColor(cv2.resize(frame, dsize=(img_sz, img_sz)),cv2.COLOR_BGR2GRAY)
            except:frame = np.zeros([img_sz,img_sz])
            batch_frames.append(frame)
        else:
            batch_list = []
            for i in range(batch_sz):
                batch_list.append(np.stack(batch_frames[i:i+3],axis=-1))
            yield np.stack(batch_list,axis=0)/127.5-1.0
            batch_frames = batch_frames[-2:]
    cap.release()

frames = 3
batch_sz = 8
img_sz = 64

feed_s = tf.placeholder(tf.float32,[batch_sz,img_sz,img_sz,frames])
feed_y = tf.placeholder(tf.float32,[batch_sz,img_sz,img_sz,frames])

w = get_UNet_params()
y = get_UNet(feed_s,w)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

sess = tf.Session()

saver.restore(sess, tf.train.get_checkpoint_state('./cpt/').model_checkpoint_path)

iter = 0

#----------------------------------------------------------------------------------------------------
def remove_noise(mask,ksize = 3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

video_names = ["Fountain.avi","input.avi","input2.avi","input3.avi","input4.avi",'people_aligned.avi']
static_names = ['input.avi','input3.avi']
varing_bg_names = ['input2.avi',"input4.avi"]
moving_bg_names = ['people.avi','people_aligned.avi']

if not os.path.exists('/home/laodar/GithubProjects/model/static/'):
    os.mkdir('/home/laodar/GithubProjects/model/static/')
if not os.path.exists('/home/laodar/GithubProjects/model/varing/'):
    os.mkdir('/home/laodar/GithubProjects/model/varing/')
if not os.path.exists('/home/laodar/GithubProjects/model/moving/'):
    os.mkdir('/home/laodar/GithubProjects/model/moving/')
if not os.path.exists('/home/laodar/GithubProjects/model/static/UNet/'):
    os.mkdir('/home/laodar/GithubProjects/model/static/UNet/')
if not os.path.exists('/home/laodar/GithubProjects/model/varing/UNet/'):
    os.mkdir('/home/laodar/GithubProjects/model/varing/UNet/')
if not os.path.exists('/home/laodar/GithubProjects/model/moving/UNet/'):
    os.mkdir('/home/laodar/GithubProjects/model/moving/UNet/')

video_name = static_names[1]
video_path = '/home/laodar/GithubProjects/model/' + video_name
save_name_prefix = '/home/laodar/GithubProjects/model/static/UNet/'+video_name.split('.')[0]+'_'
#----------------------------------------------------------------------------------------------------
'''
vit = video_iter(video_path)
vit.next()
second_batch = vit.next()
tenth_mask = sess.run(y, feed_dict={feed_s: second_batch})[0]
cv2.imwrite(save_name_prefix + '1.png',tenth_mask*255.0)
'''
path = "./people.avi"
for batch in video_iter(path):
    test_y = sess.run(y, feed_dict={feed_s: batch})
    img_y_test = np.concatenate([np.stack([test_y[j, :, :, 0]] * 3, axis=-1) for j in range(8)], axis=0)
    img_s_test = np.concatenate([batch[j, :, :, :] / 2.0 + 0.5 for j in range(8)], axis=0)
    img_show_test = np.concatenate([img_s_test, img_y_test], axis=1)
    cv2.imshow('s', img_show_test)
    save_dir = path[:-len(path.split('/')[-1])] + str(iter) + '.png'
    cv2.imwrite(save_dir,img_show_test*255.0)
    cv2.waitKey(-1)
    print iter
    iter += 1
