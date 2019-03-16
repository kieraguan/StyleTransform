import tensorflow as tf
import numpy as np

class Encoder:
    """
    Using a vgg19 pretrain model as an encoder of different feature layer
    """
    def __init__(self, pretrain_path="./vgg19.npy"):
        # load pretrained vgg19 model
        self.data_dict = np.load(pretrain_path, encoding='latin1').item()
        print("npy file loaded")

    def max_pool_layer(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = tf.constant(self.data_dict[name][0], name="filter")
            filt_size=3
            bottom = tf.pad(bottom,[[0,0],[int(filt_size/2),int(filt_size/2)],[int(filt_size/2),int(filt_size/2)],[0,0]],mode= 'REFLECT')
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            bias = tf.nn.bias_add(conv, tf.constant(self.data_dict[name][1], name="biases"))
            relu = tf.nn.relu(bias)
            return relu
    
    def encoder(self,inputs,feature_layer):
        num_layers = int(feature_layer[-1])
        
        # The number of kernels of each feature layer
        vgg_layers_par={
                        'relu1':[('conv1_1',64), ('conv1_2',64), ('pool1',64)],
                        'relu2':[('conv2_1',128), ('conv2_2',128), ('pool2',128)],
                        'relu3':[('conv3_1',256), ('conv3_2',256), ('conv3_3',256), ('conv3_4',256), ('pool3',256)],
                        'relu4':[('conv4_1',512), ('conv4_2',512), ('conv4_3',512), ('conv4_4',512), ('pool4',512)],
                        'relu5':[('conv5_1',512), ('conv5_2',512), ('conv5_3',512), ('conv5_4',512)]
        }
        
        # construct feature layers
        outputs = inputs
        for d in range(1,num_layers+1):
            for layer in vgg_layers_par["relu" + str(d)]:                
                if 'conv' in layer[0] :
                    outputs =self.conv_layer(outputs,layer[0])
                if 'pool' in layer[0] and d <num_layers :
                    outputs = self.max_pool_layer(outputs,layer[0])
        return outputs