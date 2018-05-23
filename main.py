import os
import sys
import pandas as pd
import tensorflow as tf
from DataFactory import FeatureDictionary,DataParser
import numpy as np
from model import PairWiseDeepFm
import random
import pickle
from utils import *
random.seed(123)


df_list = []
for line in file("./data/u.user"):
    line_list = line.strip().split('|')
    t_list = []
    for val in line_list:
        t_list.append(val)
    df_list.append(t_list)    
u_df = pd.DataFrame(df_list);
u_df.columns = ['uid','age','sex','occupation','zipCode']  
u_df.to_csv('./data/user_feat.csv',index = None)


i_list = []
for line in file('./data/u.item'):
    line_list = line.strip().split('|')
    t_list = []
    for val in line_list:
        t_list.append(val)
    i_list.append(t_list)
i_df = pd.DataFrame(i_list)
columns = ['iid','iname','itime','null','iwebsite']
for i in range(len(t_list)-len(columns)):
    columns.append('feat'+str(i));
i_df.columns = columns
i_df.to_csv('./data/item_feat.csv',index = None)

ignore_cols = ['zipCode','uid','iid','null','iwebsite','itime','iname']
numeric_cols = ['age']

feat_dict = FeatureDictionary(u_df,i_df,ignore_cols,numeric_cols)
dp = DataParser(feat_dict,u_df,i_df,ignore_cols,numeric_cols)

def evaltest(sess):
    liens = open('./data/movielens-100k-test.txt').readlines()
    userPosTest = pickle.load(open('./data/userTestPos.pkl','rb'))
    res = []
    for u in userPosTest.keys():
        if len(userPosTest[u]) <  10:
            continue
        user,itemp,user_feat,user_feat_val,item_feat,item_feat_val,label = dp.get_data_test(u)
        feat_catep = np.hstack((user_feat,item_feat))
        feat_val1 = np.hstack((user_feat_val,item_feat_val))
        label1 = np.reshape(label,(len(label),1))
        score = model.eval(sess,user,itemp,feat_catep,feat_val1,label1)[0]
        score = np.reshape(np.array(score),(-1,))
        score_label = zip(score,label)
        score_label = sorted(score_label, cmp=lambda x,y:cmp(x[0],y[0]),reverse = True)
        r = [x[1] for x in score_label[:10]]
        res.append(ndcg_at_k(r))
        if len(res) > 50:
            break
    return np.mean(res)
        
        

gpu_options = tf.GPUOptions(allow_growth =True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    model = PairWiseDeepFm(0.001,61,22)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    sys.stdout.flush()
    lines = open('./data/movielens-100k-train.txt','rb').readlines()
    batch_size = 32
    for d in range(50):
        random.shuffle(lines)
        epoch_size = round(len(lines) / batch_size)
        ind = 1
        time = 0
        while ind+batch_size < len(lines):
            time += 1
            user,itemp,itemn,user_feat,user_feat_val,itemp_feat,itemp_feat_val,itemn_feat,itemn_feat_val,label = dp.get_batch_data(lines,ind,batch_size)
            ind = ind + batch_size
            feat_catep = np.hstack((user_feat,itemp_feat))
            feat_val1 = np.hstack((user_feat_val,itemp_feat_val))
            feat_caten = np.hstack((user_feat,itemn_feat))
            feat_val2 = np.hstack((user_feat_val,itemn_feat_val))
            label = np.reshape(label,(len(label),1))
            loss = model.fit(sess,user,itemp,itemn,feat_catep,feat_val1,feat_caten,feat_val2,label)
            if time % 100 == 0:
                print('Epoch %d Global_step %d\tTrain_loss: %.4f' %(d,time,loss))
                sys.stdout.flush()
            if time % 300 == 0:
                print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_NDCG@10: %.4f' %(d,time,loss,evaltest(sess)))
                sys.stdout.flush()
                
            
