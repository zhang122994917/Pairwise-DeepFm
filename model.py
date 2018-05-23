import tensorflow as tf

class PairWiseDeepFm(object):
    
    def __init__(self,learning_rate,featureSize,fieldSize,userCount=943,itemCount=1682):
        
        self.fieldSize = fieldSize
        self.learning_rate = learning_rate
        self.featureSize = featureSize
        self.u = tf.placeholder(tf.int32,[None,])
        self.i1 = tf.placeholder(tf.int32,[None,])
        self.i2 = tf.placeholder(tf.int32,[None,])
        self.feat_cate1 = tf.placeholder(tf.int32,[None,None])
        self.feat_val1 = tf.placeholder(tf.float32,[None,None])
        self.feat_cate2 = tf.placeholder(tf.int32,[None,None])
        self.feat_val2 = tf.placeholder(tf.float32,[None,None])
        self.label = tf.placeholder(tf.int32,[None,1])
        Hidden_units = 128
	self.Hidden_units = Hidden_units
        self.user_emb_w = tf.get_variable('user_emb_w',[userCount,Hidden_units])
        self.item_emb_w = tf.get_variable('item_emb_w',[itemCount,Hidden_units])
        self.w_first = tf.get_variable("first_weight",(self.featureSize,1),
                                initializer = tf.random_normal_initializer(0.0,0.01))
        self.feature_emb_w =tf.get_variable('feature_emb_w',[self.featureSize,Hidden_units/2])
        self.build_model()
        
    
    def single_score(self,u,i,feat_cate,feat_val,reuse = False):
       
	feat_val = tf.reshape(feat_val, shape=[-1, self.fieldSize, 1]) 
        u_emb = tf.nn.embedding_lookup(self.user_emb_w,u) #[None,h]
	i_emb = tf.nn.embedding_lookup(self.item_emb_w,i) #[None,h]
        feature_embeddings = tf.nn.embedding_lookup(self.feature_emb_w,feat_cate)#[None,h2]
        
        #first-order
        first_emb = tf.nn.embedding_lookup(self.w_first,feat_cate)
        y_first_part = tf.reduce_sum(tf.multiply(first_emb,feat_val),2) #[None,f]
 
        #second-order
        emb = tf.multiply(feature_embeddings,feat_val)
        sum_squared_part = tf.square(tf.reduce_sum(emb,1))
        squared_sum_part = tf.reduce_sum(tf.square(emb),1)
        y_second_part = 0.5*tf.subtract(sum_squared_part,squared_sum_part) #[None * k]
 
        #fcn
        flat_emb= tf.reshape(feature_embeddings,[-1,self.fieldSize*self.Hidden_units/2])
        all_emb =tf.concat([u_emb,i_emb,flat_emb],axis = 1)

	if reuse:
        	bn_layer = tf.layers.batch_normalization(inputs = all_emb,name='bn',reuse = True)
        	layer1 = tf.layers.dense(bn_layer,128,activation = tf.nn.sigmoid,name = 'f1',reuse = True)
        	layer2 = tf.layers.dense(layer1,64,activation = tf.nn.sigmoid,name = 'f2',reuse = True)
        	layer3 = tf.layers.dense(layer2,1,activation = tf.nn.sigmoid,name = 'f3',reuse = True)
        	#deepfm
        	deep_out = tf.concat([y_first_part,y_second_part,layer3],axis = 1)
        	res_out = tf.layers.dense(deep_out,1,activation=None,name = 'f4',reuse = True)
	else:
        	bn_layer = tf.layers.batch_normalization(inputs = all_emb,name='bn')
        	layer1 = tf.layers.dense(bn_layer,128,activation = tf.nn.sigmoid,name = 'f1')
        	layer2 = tf.layers.dense(layer1,64,activation = tf.nn.sigmoid,name = 'f2')
        	layer3 = tf.layers.dense(layer2,1,activation = tf.nn.sigmoid,name = 'f3')
        	#deepfm
        	deep_out = tf.concat([y_first_part,y_second_part,layer3],axis = 1) 
        	res_out = tf.layers.dense(deep_out,1,activation=None,name = 'f4')

        return res_out
    
    
    def build_model(self):
        self.out1 = self.single_score(self.u,self.i1,self.feat_cate1,self.feat_val1)
        self.out2 = self.single_score(self.u,self.i2,self.feat_cate2,self.feat_val2,reuse=True)
        
        self.out = tf.nn.sigmoid(self.out1-self.out2)
        self.loss = tf.losses.log_loss(self.label,self.out)
        #optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)
    
        
        
    def fit(self,sess,user,itemp,itemn,feat_catep,feat_val1,feat_caten,feat_val2,label):
        loss,_ = sess.run([self.loss,self.optimizer],feed_dict={
            self.u: user,
            self.i1: itemp,
            self.i2: itemn,
            self.feat_cate1: feat_catep,
            self.feat_val1: feat_val1,
            self.feat_cate2: feat_caten,
            self.feat_val2: feat_val2,
            self.label: label,
        })
        return loss
    
    def save(self,sess,path):
        saver = tf.train.Saver()
        saver.save(sess,save_path=path)
        
    def eval(self,sess,user,itemp,feat_cate,feat_val,label):
        resScore = sess.run([self.out1],feed_dict={
            self.u:user,
            self.i1:itemp,
            self.i2:itemp,
            self.feat_cate1:feat_cate,
            self.feat_val1:feat_val,
            self.feat_cate2:feat_cate,
            self.feat_val2:feat_val,
            self.label:label,
        })
        return resScore
        
        
