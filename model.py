import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
import numpy as np
import cv2
from ops import *
from data import *
from BGGenerator import *
import data14

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

frames = 3
batch_sz = 8
img_sz = 64

feed_s = tf.placeholder(tf.float32,[batch_sz,img_sz,img_sz,frames])
feed_y = tf.placeholder(tf.float32,[batch_sz,img_sz,img_sz,frames])

w = get_UNet_params()
y = get_UNet(feed_s,w)

loss = tf.reduce_mean((y[:,:,:,0]-feed_y[:,:,:,1])**2.0)#*(feed_y[:,:,:,2]+0.3))

train = tf.train.AdamOptimizer(learning_rate=0.0005,beta1=0.9).minimize(loss)

init = tf.initialize_all_variables()

saver = tf.train.Saver()

sess = tf.Session()

sess.run(init)

saver.restore(sess, tf.train.get_checkpoint_state('./cpt/').model_checkpoint_path)

iter = 1360000

it_test = BackgroundGenerator(test_iter(frames=3))

for batch,label in BackgroundGenerator(data14.train_iter(batch_sz=batch_sz,frames=3)):
    #print np.min(batch),np.max(batch),np.min(label),np.max(label)
    print batch.shape,label.shape
    iter += 1
    y_train,ls,_ = sess.run([y,loss,train],feed_dict={feed_s:batch,feed_y:label})

    test_s,test_y= it_test.next()
    y_test = sess.run(y,feed_dict={feed_s:test_s})
    print iter,ls
    img_y_test = np.concatenate([np.stack([y_test[j, :, :, 0]]*3,axis=-1) for j in range(8)],axis=0)
    img_label_test =  np.concatenate([np.stack([test_y[j, :, :, 1]]*3,axis=-1) for j in range(8)],axis=0)
    img_s_test = np.concatenate([test_s[j, :, :, :3] / 2.0 + 0.5 for j in range(8)],axis=0)
    img_s_train = np.concatenate([batch[j, :, :, :3] / 2.0 + 0.5 for j in range(8)],axis=0)
    img_y_train = np.concatenate([np.stack([y_train[j, :, :, 0]]*3,axis=-1) for j in range(8)],axis=0)
    img_label_train = np.concatenate([np.stack([label[j, :, :, 1]]*3,axis=-1) for j in range(8)],axis=0)
    #print img_y_test.shape,img_s_test
    img_show_test = np.concatenate([img_s_test,img_y_test,img_label_test],axis=1)
    img_show_train = np.concatenate([img_s_train, img_y_train,img_label_train], axis=1)
    cv2.imshow('s',np.concatenate([img_show_test,img_show_train],axis=0))
    #cv2.imshow('s', img_show_train)
    cv2.waitKey(1)
    if iter%10000==0:
        saver.save(sess, './cpt/pretrained', global_step=iter)