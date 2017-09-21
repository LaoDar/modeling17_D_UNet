import tensorflow as tf

def maxpool2d(x,k=2):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,k,k,1],padding='SAME')

def fullyconnect(h,w,b):
    return tf.add(tf.matmul(h,w),b)


def relu(x):
    return tf.nn.relu(x)


def lrelu(x, leaky=0.2):
    return tf.maximum(x, leaky * x)


def conv2x2(x, w):
    return tf.nn.conv2d(x, w, [1, 2, 2, 1], padding='SAME')

def conv2d(x,W,b,stride=1):
    x=tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')
    x=tf.nn.bias_add(x,b)
    return lrelu(x)

def conv1x1(x, w,padding='SAME'):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], padding=padding)

'''
def deconv2x2(x, w,output_shape):
    #print output_shape
    return tf.nn.conv2d_transpose(x, w, output_shape, [1, 2, 2, 1], padding='SAME')
'''
def deconv2x2(x, w):
    input_shape = x.get_shape().as_list()
    out_ch = w.get_shape().as_list()[-2]
    output_shape = [input_shape[0], input_shape[1] * 2, input_shape[2] * 2, out_ch]
    return tf.nn.conv2d_transpose(x, w, output_shape, [1, 2, 2, 1], padding='SAME')


def deconv1x1(x, w,output_shape):
    #print output_shape
    return tf.nn.conv2d_transpose(x, w, output_shape, [1, 1, 1, 1], padding='SAME')

def sigmoid_xent(logits, targets):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits, targets)
