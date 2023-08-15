# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:00:16 2019

@author: UX501
"""
import tensorflow.contrib.image
import tensorflow as tf
import numpy as np
import sys
import os
import cv2
import h5py
import scipy.misc
from tensorflow.python.platform import gfile

# In[2]:


def main():
    path='../getmax/embedding.h5'
    if os.path.exists(path):
        os.remove(path)
        print('embedding have been removed！！！')
    embs,class_arr=cv2_face()
   
    f=h5py.File(path,'w')
    class_arr=[i.encode() for i in class_arr]
    embs = [j for j in embs]
#    all_embs = []
#    for emb in embs:
#        all_embs.append(prewhiten(emb))
    f.create_dataset('class_name',data=class_arr)
    f.create_dataset('embeddings',data=embs)
    f.close()



# In[3]:resize face
#use Haar Find face
def cv2_face():
#    cascPath = "C:/Users/UX501/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
#    faceCascade = cv2.CascadeClassifier(cascPath)
     with tf.Graph().as_default():
        with tf.Session() as sess:
            load_model('../model/') 
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("Mul:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            keep_probability_placeholder= tf.get_default_graph().get_tensor_by_name('keep_probability:0')
            path = '../../FN8_Face'
#            sess.run(tf.global_variables_initializer())
            class_names_arr=[]
            files1 = os.listdir(path)
            embs = []
            for step in files1:
                print(step)
                if step != 'embedding.h5':
                    l_image = os.path.join(path,step)
                    al_image = os.listdir(l_image)
                    count = 0
    #                s = len(al_image)
    #                if s != 1:
                    for i in al_image:
                        if count <= 1000:
                            scaled_arr=[]
                            # img = scipy.misc.imread(os.path.join(l_image,i), mode='L')
                            img = cv2.imread(os.path.join(l_image,i))
    #                        img = cv2.equalizeHist(img)
                            #get face we resize our image of face
                            scaled =cv2.resize(img,(160,160),interpolation=cv2.INTER_LINEAR)
                            scaled.astype(float)
                            scaled = prewhiten(scaled)
    #                        scaled_mean = np.mean(scaled)
    #                        scaled = scaled-scaled_mean
    #                        scaled.astype(int)
                            
                            #model is gray image model, so our shape must be (image_size,image_size,1)
                            scaled = scaled[:,:,1]
                            scaled = np.array(scaled).reshape(160, 160, 1)
                            #check shape
                            print(scaled.shape,count)
                            scaled_arr.append(scaled)
                            class_names_arr.append(step)
                            
                        
                            # calculate embeddings
                            feed_dict = { images_placeholder: scaled_arr, phase_train_placeholder:False ,keep_probability_placeholder:1.0}
        
                            embs.append(sess.run(embeddings, feed_dict=feed_dict))
                        else:
                            break
                        count = count + 1
            return embs,class_names_arr

# In[4]:


def load_model(model_dir,input_map=None):
    '''reload model'''
    ckpt = tf.train.get_checkpoint_state(model_dir)                         
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')   
    saver.restore(tf.get_default_session(), ckpt.model_checkpoint_path)
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std,1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x,mean),1/std_adj)
    return y

# In[ ]:


if __name__=='__main__':
    main()
