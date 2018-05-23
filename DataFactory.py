import random
import linecache
import pickle
random.seed(123)


userPos = pickle.load(open('./data/userPos.pkl','rb'))
userTestPos = pickle.load(open('./data/userTestPos.pkl','rb'))
class FeatureDictionary(object):
    
    def __init__(self,u_df,i_df,ignore_cols,numeric_cols):
        self.ignore_cols = ignore_cols
        self.numeric_cols = numeric_cols
        self.gen_feat_dict(u_df,i_df)
    
    #generat the user feature dict and item feature dict
    def gen_feat_dict(self,u_df,i_df):
        self.feat_dict = {}
        tc = 0
        for col in u_df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                self.feat_dict[col] = tc
            else:
                us = u_df[col].unique()
                self.feat_dict[col] = dict(zip(us,range(tc,len(us)+tc)))
                tc += len(us)
        for col in i_df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                self.feat_dict[col] = tc
            else:
                us = i_df[col].unique()
                self.feat_dict[col] = dict(zip(us,range(tc,len(us)+tc)))
                tc += len(us)
        self.feat_dim = tc
        
import linecache
class DataParser(object):
    def __init__(self,feat_dict,u_df,i_df,ignore_cols,numeric_cols):
        self.ignore_cols = ignore_cols
        self.numeric_cols = numeric_cols
        self.feat = feat_dict
        self.parse(u_df,i_df)
     
    #parse the user feature and item feature
    def parse(self,u_df,i_df):
        
        #u_df store the user feature index
        #u_df store the user feature value
        u_df_val = u_df.copy();
        for col in u_df:
            if col in self.ignore_cols:
                u_df.drop(col,axis = 1,inplace = True)
                u_df_val.drop(col,axis = 1,inplace = True)
                continue
            if col in self.numeric_cols:
                u_df[col] = self.feat.feat_dict[col]
            else:
                u_df[col] = u_df[col].map(self.feat.feat_dict[col])
                u_df_val[col] = 1.
        
        #i_df store the item feature index
        #i_df_val store the item feature value
        i_df_val = i_df.copy()
        for col in i_df:
            if col in self.ignore_cols:
                i_df.drop(col,axis = 1,inplace = True)
                i_df_val.drop(col,axis =1,inplace = True)
                continue
            if col in self.numeric_cols:
                i_df[col] = self.feat.feat_dict[col]
            else:
                i_df[col] = i_df[col].map(self.feat.feat_dict[col])
                i_df_val[col] = 1.
        self.u_df = u_df
        self.u_df_val = u_df_val
        self.i_df = i_df
        self.i_df_val = i_df_val
        return u_df,u_df_val,i_df,i_df_val
    
    # according to useid, sort the itemid based on output score and rank
    def get_data_test(self,u):
        user = []
        itemp = []
        items = set(range(1682))-set(userPos[u])
        items = random.sample(items,100)       
        itemp = list(items)
        user = [int(u) for item in itemp]
        
        user_feat = []
        user_feat_val = []
        for u in user:
            user_feat.append(self.u_df.iloc[u].values)
            user_feat_val.append(self.u_df_val.iloc[u].values)
        
        item_feat = []
        item_feat_val = []
        for i in itemp:
            item_feat.append(self.i_df.iloc[i].values)
            item_feat_val.append(self.i_df_val.iloc[i].values)
         
        label = []
        for item in itemp:
            if str(item) in userTestPos[str(u)]:
                label.append(1)
            else:
                label.append(0)
        return user,itemp,user_feat,user_feat_val,item_feat,item_feat_val,label
    
    
    # generate pairwise dataset batch
    def get_batch_data(self,lines,index,size):
        user = []
        itemp = []
        itemn = []
        label = []
        all_item = range(1682)
        #lines = open(filename,'rb').readlines()
        for i in range(index,index+size):
            line = lines[i]
            line = line.strip()
            line = line.split('\t')
            user.append(line[0])
            itemp.append(line[1])
            tmp_item = random.choice(all_item)
            while tmp_item in userPos[line[0]]:
                tmp_item = random.choice(all_item)
            itemn.append(tmp_item)
            label.append(1)
        
        user = [int(x) for x in user]
        itemp = [int(x) for x in itemp]
        itemn = [int(x) for x in itemn]
        user_feat = []
        user_feat_val = []
        for u in user:
            user_feat.append(self.u_df.iloc[u].values)
            user_feat_val.append(self.u_df_val.iloc[u].values)
        
        itemp_feat = []
        itemp_feat_val = []
        for i in itemp:
            itemp_feat.append(self.i_df.iloc[i].values)
            itemp_feat_val.append(self.i_df_val.iloc[i].values)
            
        itemn_feat = []
        itemn_feat_val = []
        for i in itemn:
            itemn_feat.append(self.i_df.iloc[i].values)
            itemn_feat_val.append(self.i_df_val.iloc[i].values)
        
        return user,itemp,itemn,user_feat,user_feat_val,itemp_feat,itemp_feat_val,itemn_feat,itemn_feat_val,label
                
