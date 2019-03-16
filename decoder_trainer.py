import tensorflow as tf
import numpy as np
from skimage.io import imsave,imshow,imread
import utils as utils
from encoder import Encoder
from decoder import Decoder
from wct import wct

class DecoderTrainer:
    def __init__(self, target_layer=None, 
                 pretrained_path=None, 
                 epoch = 1, 
                 checkpoint_path=None, 
                 batch_size=None, 
                 loss_lambda = 1, 
                 pixel_loss_hist = [], 
                 feature_loss_hist = []):
        self.pretrained_path = pretrained_path
        self.target_layer = target_layer
        self.encoder = Encoder(self.pretrained_path)
        self.epoch = epoch
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.loss_lambda = 1
        self.pixel_loss_hist = pixel_loss_hist
        self.feature_loss_hist = feature_loss_hist
        
    def encoder_decoder(self,inputs):
        encoded = self.encoder.encoder(inputs,self.target_layer)
        model=Decoder()
        decoded,_ = model.decoder(encoded,self.target_layer)
        decoded_encoded= self.encoder.encoder(decoded,self.target_layer)
        
        return encoded,decoded,decoded_encoded
    
    
    def train(self, pre_trained_model = None):
        inputs = tf.placeholder('float',[None,224,224,3])
        outputs = tf.placeholder('float',[None,224,224,3])
        
        encoded,decoded,decoded_encoded = self.encoder_decoder(inputs)
        
        pixel_loss = tf.losses.mean_squared_error(decoded,outputs)
        feature_loss = tf.losses.mean_squared_error(decoded_encoded,encoded)
        loss = pixel_loss+ self.loss_lambda * feature_loss
        opt= tf.train.AdamOptimizer(0.0001).minimize(loss)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config = config)as sess  :
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)  

            saver = tf.train.Saver()

            if pre_trained_model is not None:
                try:
                    print("Load the model from: {}".format(pre_trained_model))
                    saver.restore(sess, 'models/{}'.format(pre_trained_model))
                except Exception:
                    raise ValueError("Load model Failed!")
            
            
            current_loss = 100000000
            iter_total = 0
            group_size = 1000
            iters = int(group_size / self.batch_size)
            for epc in range(self.epoch):
                print("epoch {} ".format(epc + 1))
                for group_idx in range(int(60000 / group_size)):
                    input_data = utils.load_images("../train2014", (group_size, 224,224,3), group_idx * group_size)
                    print("Group data: ", group_idx * group_size)
                    for itr in range(iters):
                        iter_total += 1
                        batch_x = input_data[itr * self.batch_size: (1 + itr) * self.batch_size]

                        feed_dict = {inputs:batch_x, outputs : batch_x}
                        _,p_loss,f_loss,reconstruct_imgs=sess.run([opt,pixel_loss,
                                                   feature_loss,
                                                   decoded],
                                                   feed_dict=feed_dict)
                        self.pixel_loss_hist.append(p_loss)
                        self.feature_loss_hist.append(f_loss)
                        if iter_total % 5 ==0:
                            print('step %d | pixel_loss is %f | feature_loss is %f |'%(iter_total,p_loss,f_loss))
                            result_img = np.clip(reconstruct_imgs[0],0,255).astype(np.uint8)
                            imsave('recover_images/result{:0>5d}.jpg'.format(iter_total),result_img)
                            if p_loss + self.loss_lambda * f_loss <= current_loss:
                                current_loss = p_loss + self.loss_lambda * f_loss
                                print("best loss: ", current_loss)
                                saver.save(sess,self.checkpoint_path)

            coord.request_stop()  
            coord.join(threads)
        return self.pixel_loss_hist, self.feature_loss_hist
             

            
        



       
             

    
        
