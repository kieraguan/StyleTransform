import tensorflow as tf

import numpy as np

class Decoder:
    """
    The decoder is a reversed vgg network which is used to construct the image 
    from features of corresponding layer.
    """
    def __init__(self, pretrain_path=None, is_training=True, dropout=0.5):
        if pretrain_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None
        self.var_dict = {}
        self.is_training = is_training
        self.dropout = dropout
        
    def upsample(self,bottom,height):
        new_height=height*2
        return tf.image.resize_images(bottom, [new_height, new_height], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    def get_var(self, initial_value, name, idx, var_name):
        """
        Restore data from pretrained model or initialize new variables
        """
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
            print ('resore %s weight'%(name))
        else:
            value = initial_value

        if self.is_training:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        assert var.get_shape() == initial_value.get_shape()

        return var
    
    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases   
    
    def conv_layer(self, bottom, in_channels, out_channels, name,var_list,is_training=True):
        filt_size = 3
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(filt_size, in_channels, out_channels, name)
            
            bottom = tf.pad(bottom,[[0,0],[int(filt_size/2),int(filt_size/2)],[int(filt_size/2),int(filt_size/2)],[0,0]],mode= 'REFLECT')
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)  
            
            var_list.append(filt)
            var_list.append(conv_biases)
            return relu,var_list
        
    def output_layer(self, bottom, in_channels, out_channels, name,var_list):
        with tf.variable_scope(name):
            filt_size = 9
            filt, conv_biases = self.get_conv_var(filt_size, in_channels, out_channels, name)
            bottom = tf.pad(bottom,[[0,0],[int(filt_size/2),int(filt_size/2)],[int(filt_size/2),int(filt_size/2)],[0,0]],mode= 'REFLECT')
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            bias = tf.nn.bias_add(conv, conv_biases)
            
            var_list.append(filt)
            var_list.append(conv_biases)
            return bias,var_list

    def decoder(self,inputs,feature_layer) :
        num_layers = int(feature_layer[-1])
        var_list=[]
        vgg_layers_par={
                'relu5':[('dconv5_1',512,512), ('dconv5_2',512,512), ('dconv5_3',512,512), ('dconv5_4',512,512)],
                'relu4':[('upsample',28,56), ('dconv4_1',512,256), ('dconv4_2',256,256), ('dconv4_3',256,256), ('dconv4_4',256,256)],
                'relu3':[('upsample',56,112), ('dconv3_1',256,128), ('dconv3_2',128,128), ('dconv3_3',128,128), ('dconv3_4',128,128)],
                'relu2':[('upsample',112,224), ('dconv2_1',128,64), ('dconv2_2',64,64)],
                'relu1':[('dconv1_1',64,64),('output',64,3)]} 
        
        outputs = inputs
        for d in reversed(range(1,num_layers+1)):
            for layer in vgg_layers_par["relu" + str(d)]:
                if 'up' in layer[0]:
                    outputs = self.upsample(outputs,layer[1])
                if 'dconv' in layer[0] :
                    outputs ,var_list= self.conv_layer(outputs,layer[1],layer[2],layer[0]+'_'+ feature_layer,var_list)
                if 'out' in layer[0] :
                    outputs, var_list = self.output_layer(outputs,layer[1],layer[2],layer[0]+'_'+feature_layer,var_list)
                    
        return outputs , var_list
