#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def wct(content, style, alpha, eps=1e-8):
    #reshape H*W*C to C*H*W
    c_t = np.transpose(np.squeeze(content),(2,0,1))
    s_t = np.transpose(np.squeeze(style),(2,0,1))
    Cc,Hc,Wc=c_t.shape
    Cs,Hs,Ws=s_t.shape
    #reshape to C*(H*W)
    content=np.reshape(c_t,(Cc,Hc*Wc))
    style=np.reshape(s_t,(Cs,Hs*Ws))
    #fc-mean
    mc=np.mean(content, axis=1, keepdims=True)
    fc=content-mc
    #covariance
    conv_c=np.cov(fc,rowvar=True)
    #fs-mean
    ms=np.mean(style, axis=1,keepdims=True)
    fs=style-ms
    #conv_s=np.matmul(fs,np.transpose(fs))/(Hs*Ws-1)+np.identity(Cs)*eps
    conv_s=np.cov(fs,rowvar=True)
    #get Cu: matrix of eigenvectors,Cs: eigenvalues
    Cu,Cs,Cv=np.linalg.svd(conv_c)
    Dc=np.diag(1.0/np.sqrt(Cs+eps))
    #calculation of fc_hat=Cu*Dc*Cu.T*fc
    fc_hat=np.dot(np.dot(np.dot(Cu,Dc),Cu.T),fc)
    Su,Ss,Sv=np.linalg.svd(conv_s)
    Ds=np.diag(np.sqrt(Ss+eps))
    fcs_hat=np.dot(np.dot(np.dot(Su,Ds),Su.T),fc_hat)
    #recenter fcs_hat
    fcs_hat+=ms
    result=alpha*fcs_hat+(1-alpha)*(fc+mc)
    #shape it back to H*W*C
    result_img=np.reshape(result,(Cc,Hc,Wc))
    img_w = np.transpose(result_img,(1,2,0))
    img_w = np.expand_dims(img_w, axis = 0)
    return img_w