import tensorflow as tf
import numpy as np
from skimage.io import imsave
import utils
from encoder import Encoder
from decoder import Decoder
from wct import wct

class StyleTransferrer:
    def __init__(self,content_path,style_path,alpha,pretrained_vgg,output_path) :
        self.content_path = content_path
        self.style_path = style_path
        self.output_path = output_path
        self.alpha = alpha
        self.encoder = Encoder(pretrained_vgg)
        self.decoder = Decoder()  
        self.decoder_weights = ['models/decoder_1.ckpt','models/decoder_2.ckpt','models/decoder_3.ckpt','models/decoder_4.ckpt', 'models/decoder_5.ckpt']
        self.encoder_out_size = [[1, 224, 224, 64],[1, 112, 112, 128],[1, 56, 56, 256], [1, 28, 28, 512], [1,14,14,512]]
    
    def transfer(self, num_layers = 5, reverse = False):
        content = tf.placeholder('float',[1,224,224,3])
        style = tf.placeholder('float',[1,224,224,3])
        
        
        
        content_encode = []
        style_encode = []
        blended = []
        stylized = []
        var_list = []
        saver = []
        layer_input = content
        # construct the transfer network
        for i in range(num_layers):
            content_encode.append(self.encoder.encoder(content,'relu{}'.format(i+1)))
            style_encode.append(self.encoder.encoder(style,'relu{}'.format(i+1)))
            blended.append(tf.placeholder('float',self.encoder_out_size[i]))
            stylized_out ,var_list_out= self.decoder.decoder(blended[i],
                                                             'relu{}'.format(i+1))
            stylized.append(stylized_out)
            var_list.append(var_list_out)
            saver.append(tf.train.Saver(var_list_out))
        
        with tf.Session()as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            # load decoder weights
            for i in range(num_layers):
                saver[i].restore(sess,self.decoder_weights[i])
            
            # load images
            content_img = utils.load_image(self.content_path, (224,224))
            content_img = np.expand_dims(content_img,axis=0)
            style_img = utils.load_image(self.style_path, (224,224))
            style_img = np.expand_dims(style_img,axis=0)

            feed_dict = {content : content_img , style : style_img}
            
            # transfer the images
            if not reverse:
                for i in reversed(range(num_layers)):
                    # e: encoded
                    content_e, style_e = sess.run([content_encode[i], 
                                         style_encode[i]],
                                         feed_dict= feed_dict)
                    blended_e = wct(content_e,style_e, self.alpha)
                    result = sess.run([stylized[i]],feed_dict= {blended[i]: blended_e})
                    result = result[0][0]
                    result = np.expand_dims(result,axis=0)
                    feed_dict = {content : result , style : style_img}
                    result = np.squeeze(result)
                    result = np.clip(result,0,255)/255.
                    imsave(self.output_path + "relu_{}_1.jpg".format(i+1), result)
                    print("save to ", self.output_path + "relu_{}_1.jpg".format(i+1))
            else:
                for i in range(num_layers):
                    # e: encoded
                    content_e, style_e = sess.run([content_encode[i], 
                                         style_encode[i]],
                                         feed_dict= feed_dict)
                    blended_e = wct(content_e,style_e, self.alpha)
                    result = sess.run([stylized[i]],feed_dict= {blended[i]: blended_e})
                    result = result[0][0]
                    result = np.expand_dims(result,axis=0)
                    feed_dict = {content : result , style : style_img}
                    result = np.squeeze(result)
                    result = np.clip(result,0,255)/255.
                    imsave(self.output_path + "relu_{}_1.jpg".format(i+1), result) 
                    
                    
    def transfer_one_out(self, num_layers = 5, reverse = False):
        content = tf.placeholder('float',[1,224,224,3])
        style = tf.placeholder('float',[1,224,224,3])
        
        
        
        content_encode = []
        style_encode = []
        blended = []
        stylized = []
        var_list = []
        saver = []
        layer_input = content
        # construct the transfer network
        for i in range(num_layers):
            content_encode.append(self.encoder.encoder(content,'relu{}'.format(i+1)))
            style_encode.append(self.encoder.encoder(style,'relu{}'.format(i+1)))
            blended.append(tf.placeholder('float',self.encoder_out_size[i]))
            stylized_out ,var_list_out= self.decoder.decoder(blended[i],
                                                             'relu{}'.format(i+1))
            stylized.append(stylized_out)
            var_list.append(var_list_out)
            saver.append(tf.train.Saver(var_list_out))
        
        with tf.Session()as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            # load decoder weights
            for i in range(num_layers):
                saver[i].restore(sess,self.decoder_weights[i])
            
            # load images
            content_img = utils.load_image(self.content_path, (224,224))
            content_img = np.expand_dims(content_img,axis=0)
            style_img = utils.load_image(self.style_path, (224,224))
            style_img = np.expand_dims(style_img,axis=0)

            feed_dict = {content : content_img , style : style_img}
            
            # transfer the images
            if not reverse:
                for i in reversed(range(num_layers)):
                    # e: encoded
                    content_e, style_e = sess.run([content_encode[i], 
                                         style_encode[i]],
                                         feed_dict= feed_dict)
                    blended_e = wct(content_e,style_e, self.alpha)
                    result = sess.run([stylized[i]],feed_dict= {blended[i]: blended_e})
                    result = result[0][0]
                    result = np.expand_dims(result,axis=0)
                    feed_dict = {content : result , style : style_img}
                    result = np.squeeze(result)
                    result = np.clip(result,0,255)/255.
            else:
                for i in range(num_layers):
                    # e: encoded
                    content_e, style_e = sess.run([content_encode[i], 
                                         style_encode[i]],
                                         feed_dict= feed_dict)
                    blended_e = wct(content_e,style_e, self.alpha)
                    result = sess.run([stylized[i]],feed_dict= {blended[i]: blended_e})
                    result = result[0][0]
                    result = np.expand_dims(result,axis=0)
                    feed_dict = {content : result , style : style_img}
                    result = np.squeeze(result)
                    result = np.clip(result,0,255)/255.
            imsave(self.output_path + "relu_{}_1.jpg".format(num_layers), result) 

class SingleLayerTransferrer:
    def __init__(self,target_layer,content_path,style_path,alpha,pretrained_vgg,output_path,decoder_weights) :
        self.target_layer = target_layer
        self.content_path = content_path
        self.style_path = style_path
        self.output_path = output_path
        self.alpha = alpha
        self.encoder = Encoder(pretrained_vgg)
        self.decoder = Decoder()  
        self.decoder_weights = decoder_weights
        self.encoder_out_size = [[1, 224, 224, 64],[1, 112, 112, 128],[1, 56, 56, 256], [1, 28, 28, 512], [1,14,14,512]]
        
    def transfer(self):
        content = tf.placeholder('float',[1,224,224,3])
        style = tf.placeholder('float',[1,224,224,3])
       
        content_encode = self.encoder.encoder(content,self.target_layer)
        style_encode = self.encoder.encoder(style,self.target_layer)
        num_layer = int(self.target_layer[-1]) - 1
        blended = tf.placeholder('float',self.encoder_out_size[num_layer])
        
        stylized = self.decoder.decoder(blended,self.target_layer)
        saver = tf.train.Saver()
        
        with tf.Session()as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            saver.restore(sess,self.decoder_weights)           
            
            # load images
            content_img = utils.load_image(self.content_path, (224,224))
            content_img = np.expand_dims(content_img,axis=0)
            style_img = utils.load_image(self.style_path, (224,224))
            style_img = np.expand_dims(style_img,axis=0)  

            feed_dict = {content : content_img , style : style_img}

            content_e,style_e = sess.run([content_encode,style_encode],feed_dict= feed_dict)
            blended_e = wct(content_e,style_e,self.alpha)
            
            result = sess.run([stylized],feed_dict= {blended: blended_e})
            result = result[0][0]
            result = np.clip(result,0,255)/255.
            #print(e)
            result = np.squeeze(result) 
            imsave(self.output_path + ".jpg", result)