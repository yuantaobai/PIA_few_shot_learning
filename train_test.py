import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler
import torch.nn as nn
from easyfsl.utils import compute_prototypes
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
import pandas as pd
import numpy as np


def generate_csv(seg=False):
    #return 3 list:train_folder,vla_folder,test_folder
    #each list contains name of the specific class of the dog
    #eg: train_folder=['data/sp/tain/n02085620-Chihuahua',...,...]
    train_class_name_list=[]
    val_class_name_list=[]
    test_class_name_list=[]
    train_img_name_list=[]
    val_img_name_list=[]
    test_img_name_list=[]
    
    if (seg):#if use segmentation data
      train_folder = "data/seg/train"
      test_folder = "data/seg/test"
      val_folder = "data/seg/val"
    else:
      train_folder = "data/seg/train"
      test_folder = "data/seg/test"                
      val_folder = "data/seg/val"
    
    
    for train_class_name in os.listdir(train_folder):
      for train_img_name in os.listdir(os.path.join(train_folder,train_class_name)):
        train_class_name_list.append(train_class_name)
        train_img_name_list.append(os.path.join(os.path.join('train',train_class_name),train_img_name))
    df_train=pd.DataFrame({'category_name':'train','class_name':train_class_name_list,'img_name':train_img_name_list})
      
    for test_class_name in os.listdir(test_folder):
      for test_img_name in os.listdir(os.path.join(test_folder,test_class_name)):
        test_class_name_list.append(test_class_name)
        test_img_name_list.append(os.path.join(os.path.join('test',test_class_name),test_img_name))
    df_test=pd.DataFrame({'category_name':'test','class_name':test_class_name_list,'img_name':test_img_name_list})
      
    for val_class_name in os.listdir(val_folder):
      for val_img_name in os.listdir(os.path.join(val_folder,val_class_name)):
        val_class_name_list.append(val_class_name)
        val_img_name_list.append(os.path.join(os.path.join('val',val_class_name),val_img_name))
    df_val=pd.DataFrame({'category_name':'val','class_name':val_class_name_list,'img_name':val_img_name_list})
    
    return (df_train,df_test,df_val)

def get_task(df,n_way,m_shot,t_query):
    #generate task
    #input: df:dataframe ie:df_train
    #       n_way: how many class in each task
    #       m_shot: use how many img in each class as support set
    #       t_query: use how many img in each class to test the model
    #output: df_support_set: dataframe of support_set,
    #                        columns=['category_name','class_name','img_name'],
    #                        len=(n_way*m_shot)
    #
    #        df_query_set: dataframe of query_set,
    #                        columns=['category_name','class_name','img_name'],
    #                        len=(n_way*t_query)
    #
    #        class_name_dict: dictionary from classname to class_number
    df_support_set=pd.DataFrame(columns=['category_name','class_name','img_name'])
    df_query_set=pd.DataFrame(columns=['category_name','class_name','img_name'])
    class_name_dict={}
    class_name_list=df.class_name.sample(n_way).values#randomly sample n_way different class
    
    for class_num in range(n_way):
        class_name=class_name_list[class_num]
        #print(class_name,class_num)
        class_name_dict[class_name]=class_num
        #print('mshot+tquery:',m_shot+t_query)
        df_class_n=df[df['class_name']==class_name]
        #print('class n len:',len(df_class_n))
        df_class_n_random_pick=df_class_n.sample((m_shot+t_query))
        #df_class_n=df[df['class_name']==class_name].sample((m_shot+t_query))
        df_support_set=pd.concat([df_support_set,df_class_n_random_pick.head(m_shot)],axis=0,ignore_index=True)
        df_query_set=pd.concat([df_query_set,df_class_n_random_pick.tail(t_query)],axis=0,ignore_index=True)
    #df_support_set=df_support_set.reset_index()
    #df_query_set=df_query_set.reset_index() 
    return (df_support_set,df_query_set,class_name_dict)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, class_name_dict, root_dir='data/split_class',transform = None):
        self.df = df
        self.transform = transform
        self.class2index = class_name_dict
        self.rootdir=root_dir

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.iloc[index].img_name
        img_dir=os.path.join(self.rootdir,filename)
        label = self.class2index[self.df.iloc[index].class_name]
        image = Image.open( img_dir)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def set_transform(normalizing=True):
    if(normalizing):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        #resizing =  transforms.Resize(224, interpolation=2)
        transform_method=transforms.Compose([transforms.ToTensor(),normalize])
    else:
        transform_method=transforms.ToTensor()
    return transform_method

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h


#*********************************Defining networks*******************************


import torchvision.models as models
class PreTrainedResNet(nn.Module):
  def __init__(self):
    super(PreTrainedResNet, self).__init__()
    
    self.resnet = models.resnet34(pretrained=True)#resnet 18 is also fine
    #models.resnet34
    #Set gradients to false
    
    for param in self.resnet.parameters():
        param.requires_grad = False
    
    num_feats = self.resnet.fc.in_features
    self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))

    
  def forward(self, x):
    x = self.resnet (x)
    x = x.view(x.shape[0],32,4,4)
    return x

def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

    
class ProtoNet(nn.Module):
    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


class RelationNetwork(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(64,32,kernel_size=3,padding=1),  #padding
                        nn.BatchNorm2d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(32,32,kernel_size=3,padding=1),  #padding
                        nn.BatchNorm2d(32, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size,hidden_size)  #3*3: size e ax bade 2 layers
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        #out = self.fc2(out)
        return out


def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):

    def __init__(self, x_dim=1, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
            nn.Linear(z_dim,1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)


torch.cuda.empty_cache()
FEATURE_DIM = 32
RELATION_DIM = 8
RN_encoder = PreTrainedResNet()
#feature_encoder.apply(weights_init)
RN_encoder.cuda(0)

relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)
relation_network.apply(weights_init)
relation_network.cuda(0)

"""#Optimizers"""

feature_encoder_optim = torch.optim.Adam(RN_encoder.parameters(),lr=0.0005)
feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=0.0005)
relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

#proto_network_optim = torch.optim.Adam(relation_network.parameters(),lr=0.001)
#proto_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)
#nn.cont

"""#5way-5shot

For proto network
"""

(df_train,df_test,df_val)=generate_csv()

df_train.to_csv('train.csv')#saving dataset split history
df_test.to_csv('test.csv')
df_val.to_csv('val.csv')


#train relationnet and resnet
N_WAY=5
M_SHOT=5#       set to 1 if running 5way1shot
T_QUERY=5
T_QUERY_VAL=5
GPU=0
RESNET_OUT_H=4
print("Training Resnet and Relationnet...")

last_accuracy = 0.0
loss_list=[]


(df_train,df_test,df_val)=generate_csv()
for episode in range(4000):
    
    df_support_set,df_query_set,class_name_dict=get_task(df=df_train,n_way=N_WAY,m_shot=M_SHOT,t_query=T_QUERY)
    
    transform_method=set_transform(normalizing=True)
  
    support_dataset=CustomDataset(df=df_support_set,class_name_dict=class_name_dict,transform=transform_method)
    query_dataset=CustomDataset(df=df_query_set,class_name_dict=class_name_dict,transform=transform_method)
  
  
  
    sample_dataloader=DataLoader(dataset=support_dataset,batch_size=N_WAY*M_SHOT,shuffle=False)
    batch_dataloader=DataLoader(dataset=query_dataset,batch_size=N_WAY*T_QUERY,shuffle=True)
    

    # sample datas
    samples,sample_labels = sample_dataloader.__iter__().next() 
    batches,batch_labels = batch_dataloader.__iter__().next()



    # calculate features
    sample_features = RN_encoder(Variable(samples).cuda(GPU)) 
    sample_features = sample_features.view(N_WAY,M_SHOT,FEATURE_DIM,4,4)
    sample_features = torch.sum(sample_features,1).squeeze(1)
    batch_features = RN_encoder(Variable(batches).cuda(GPU)) 
    
    # calculate relations

    sample_features_ext = sample_features.unsqueeze(0).repeat(T_QUERY*N_WAY,1,1,1,1)
    batch_features_ext = batch_features.unsqueeze(0).repeat(N_WAY,1,1,1,1)
    batch_features_ext = torch.transpose(batch_features_ext,0,1)
    relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,4,4)
    relations = relation_network(relation_pairs).view(-1,N_WAY)

    mse = nn.MSELoss().cuda(GPU)
    crossen = nn.CrossEntropyLoss().cuda(0)
    one_hot_labels = Variable(torch.zeros(T_QUERY*N_WAY, N_WAY).scatter_(1, batch_labels.view(-1,1), 1).cuda(GPU))
    loss = crossen(relations,one_hot_labels)
    loss_list.append(float(loss.data.to('cpu')))

    # training

    RN_encoder.zero_grad()
    relation_network.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(RN_encoder.parameters(),0.5)
    torch.nn.utils.clip_grad_norm_(relation_network.parameters(),0.5)

    feature_encoder_optim.step()
    relation_network_optim.step()

    feature_encoder_scheduler.step(episode)
    relation_network_scheduler.step(episode)


    if (episode+1)%100 == 0:
      print("episode:",episode+1,"loss",loss.item())

    if (episode+1)%200 == 0:

        # test
        print("Testing...")
        accuracies = []
        for i in range(20):
            total_rewards = 0
            #counter = 0       #remove
            df_support_set,df_query_set,class_name_dict=get_task(df=df_val,n_way=N_WAY,m_shot=M_SHOT,t_query=T_QUERY_VAL)
    
            transform_method=set_transform(normalizing=True)#preprocess of the data, usually are normalizing+totensor
  
            support_dataset=CustomDataset(df=df_support_set,class_name_dict=class_name_dict,transform=transform_method)
            query_dataset=CustomDataset(df=df_query_set,class_name_dict=class_name_dict,transform=transform_method)
  
  
  
            sample_dataloader=DataLoader(dataset=support_dataset,batch_size=N_WAY*M_SHOT,shuffle=False)
            test_dataloader=DataLoader(dataset=query_dataset,batch_size=N_WAY*T_QUERY,shuffle=False)
    

            sample_images,sample_labels = sample_dataloader.__iter__().next()
            for test_images,test_labels in test_dataloader:
                batch_size = test_labels.shape[0]#N_WAY*T_QUERY_VAL
                # calculate features
                sample_features = RN_encoder(Variable(sample_images).cuda(GPU)) # 5x64
                sample_features = sample_features.view(N_WAY,M_SHOT,FEATURE_DIM,4,4)
                sample_features = torch.sum(sample_features,1).squeeze(1)
                test_features = RN_encoder(Variable(test_images).cuda(GPU)) # 20x64

                # calculate relations
                # each batch sample link to every samples to calculate relations
                # to form a 100x128 matrix for relation network
                sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)

                test_features_ext = test_features.unsqueeze(0).repeat(1*N_WAY,1,1,1,1)
                test_features_ext = torch.transpose(test_features_ext,0,1)
                relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,4,4)
                relations = relation_network(relation_pairs).view(-1,N_WAY)
                #print(relations)
                _,predict_labels = torch.max(relations.data,1)
                #print('predict:',predict_labels)
                #print('reallab:',test_labels)

                rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

                total_rewards = np.sum(rewards)
                #print(total_rewards,'in',len(rewards))
                #counter +=batch_size   #remove


            accuracy = total_rewards/(batch_size)
            #accuracy = total_rewards/1.0/counter
            accuracies.append(accuracy)


        test_accuracy,h = mean_confidence_interval(accuracies)

        print("val accuracy:",test_accuracy,"h:",h)

        if test_accuracy > last_accuracy:

            # save networks
            #torch.save(feature_encoder.state_dict(),str("miniimagenet/models/new/miniimagenet_feature_encoder2_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
            #torch.save(relation_network.state_dict(),str("miniimagenet/models/new/miniimagenet_relation_network2_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

            #print("save networks for episode:",episode)

            last_accuracy = test_accuracy

import matplotlib.pyplot as plt
y=loss_list
plt.plot(loss_list,'b')
plt.xlabel('number of task(iteration)')
plt.ylabel('MSE loss')
plt.title('Meta-learning loss curve for {} way {} shot'.format(N_WAY,M_SHOT))
plt.legend()
plt.show()

#test relation net
print("Testing line1 Relationnet seperately")
accuracies = []
for i in range(100):
    total_rewards = 0
    #counter = 0       #remove
    df_support_set,df_query_set,class_name_dict=get_task(df=df_val,n_way=N_WAY,m_shot=M_SHOT,t_query=T_QUERY_VAL)
    
    transform_method=set_transform(normalizing=True)#preprocess of the data, usually are normalizing+totensor
  
    support_dataset=CustomDataset(df=df_support_set,class_name_dict=class_name_dict,transform=transform_method)
    query_dataset=CustomDataset(df=df_query_set,class_name_dict=class_name_dict,transform=transform_method)
  
  
  
    sample_dataloader=DataLoader(dataset=support_dataset,batch_size=N_WAY*M_SHOT,shuffle=False)
    test_dataloader=DataLoader(dataset=query_dataset,batch_size=N_WAY*T_QUERY,shuffle=False)
    

    sample_images,sample_labels = sample_dataloader.__iter__().next()
    for test_images,test_labels in test_dataloader:
        batch_size = test_labels.shape[0]#N_WAY*T_QUERY_VAL
        # calculate features
        sample_features = RN_encoder(Variable(sample_images).cuda(GPU)) # 5x64
        sample_features = sample_features.view(N_WAY,M_SHOT,FEATURE_DIM,4,4)
        sample_features = torch.sum(sample_features,1).squeeze(1)
        test_features = RN_encoder(Variable(test_images).cuda(GPU)) # 20x64

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)

        test_features_ext = test_features.unsqueeze(0).repeat(1*N_WAY,1,1,1,1)
        test_features_ext = torch.transpose(test_features_ext,0,1)
        relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,4,4)
        relations = relation_network(relation_pairs).view(-1,N_WAY)
        #print(relations)
        #print(relations)
        _,predict_labels = torch.max(relations.data,1)
        #print('predict:',predict_labels)
        #print('reallab:',test_labels)

        rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

        total_rewards = np.sum(rewards)
        #print(total_rewards,'in',len(rewards))
        #counter +=batch_size   #remove


        accuracy = total_rewards/(batch_size)
        #accuracy = total_rewards/1.0/counter
        accuracies.append(accuracy)


test_accuracy,h = mean_confidence_interval(accuracies)
print("val accuracy:",test_accuracy,"h:",h)




#***************************define pixel similarity class**************************************
RESNET_OUT_H=4
import torchvision.models as models

class InnerproductSimilarity(nn.Module):
    def __init__(self, CLASS_NUM = N_WAY, shots = M_SHOT, metric='cosine',resnetoutsize=RESNET_OUT_H):
        super().__init__()
        self.n_way = CLASS_NUM
        self.k_shot = shots
        self.metric = metric
        self.cs=nn.CosineSimilarity(dim=2)
        
        #self.cs=nn.PairwiseDistance()
        
        self.findmaxNotrain=nn.AvgPool3d(kernel_size=(shots,4,4),stride=(shots,4,4))
        self.softmax=nn.Softmax(dim=1)
        
    def forward(self, support_features,query_features):
        numOfSupport=support_features.size(dim=0)
        numOfQuery=query_features.size(dim=0)
        support_5d=support_features.unsqueeze(0)
        support_5d_match=support_5d.repeat(numOfQuery,1,1,1,1)
        query_5d=query_features.unsqueeze(1)
        query_5d_match=query_5d.repeat(1,numOfSupport,1,1,1)#50,25,32,4,4
        cos_similarity=self.cs(support_5d_match,query_5d_match)#50,25,4,4
        score=self.findmaxNotrain(cos_similarity)#50,5每张test图片在每一类中的夹角均值
        prob=self.softmax(score)
        return score

#test pixel wise
N_WAY=5
M_SHOT=5
T_QUERY=5

PX_encoder = PreTrainedResNet()
#feature_encoder.apply(weights_init)
PX_encoder.cuda(0)


pixelsimilarity_network = InnerproductSimilarity(CLASS_NUM = N_WAY,shots = M_SHOT, metric='cosine',resnetoutsize=RESNET_OUT_H)
pixelsimilarity_network.apply(weights_init)
pixelsimilarity_network.cuda(0)


print("Testing line2 seperately...")
accuracies = []
for i in range(100):
  total_rewards = 0
  df_support_set,df_query_set,class_name_dict=get_task(df=df_val,n_way=N_WAY,m_shot=M_SHOT,t_query=T_QUERY)
  
  transform_method=set_transform(normalizing=True)#preprocess of the data, usually are normalizing+totensor
  
  support_dataset=CustomDataset(df=df_support_set,class_name_dict=class_name_dict,transform=transform_method)
  query_dataset=CustomDataset(df=df_query_set,class_name_dict=class_name_dict,transform=transform_method)
  
  
  
  support_dataloader=DataLoader(dataset=support_dataset,batch_size=N_WAY*M_SHOT,shuffle=False)
  query_dataloader=DataLoader(dataset=query_dataset,batch_size=N_WAY*T_QUERY,shuffle=False)
  

  support_images,support_labels = support_dataloader.__iter__().next()
  for test_images,test_labels in query_dataloader:
    #print(test_labels)
    support_features = PX_encoder(Variable(support_images).cuda(GPU)) # 5x64
                
    test_features = PX_encoder(Variable(test_images).cuda(GPU)) # 20x64

               
    
    relations = pixelsimilarity_network(support_features,test_features)

    _,predict_labels = torch.max(relations.data,dim=1)
    

    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(N_WAY*T_QUERY)]
    #print('correct:{}in{}'.format(sum(rewards),len(rewards)))

    total_rewards += np.sum(rewards)
    #counter +=batch_size   #remove


  accuracy = total_rewards/(1.0*N_WAY*T_QUERY)
  #accuracy = total_rewards/1.0/counter
  accuracies.append(accuracy)


test_accuracy,h = mean_confidence_interval(accuracies)

print("line2 test accuracy:",test_accuracy,"h:",h)




a=1
b=0.3
c=0.15

accuracies = []
TASK_NUM=100

#test relation net
print("Testing assemble model with a={},b={},c={}".format(a,b,c))
accuracies = []
for i_task in range(TASK_NUM):
    total_rewards = 0
    #counter = 0       #remove
    df_support_set,df_query_set,class_name_dict=get_task(df=df_val,n_way=N_WAY,m_shot=M_SHOT,t_query=T_QUERY_VAL)
    
    transform_method=set_transform(normalizing=True)#preprocess of the data, usually are normalizing+totensor
  
    support_dataset=CustomDataset(df=df_support_set,class_name_dict=class_name_dict,root_dir='data/split_class',transform=transform_method)
    query_dataset=CustomDataset(df=df_query_set,class_name_dict=class_name_dict,root_dir='data/split_class',transform=transform_method)
  
  
  
    sample_dataloader=DataLoader(dataset=support_dataset,batch_size=N_WAY*M_SHOT,shuffle=False)
    test_dataloader=DataLoader(dataset=query_dataset,batch_size=N_WAY*T_QUERY,shuffle=False)
    

    sample_images,sample_labels = sample_dataloader.__iter__().next()
    test_images,test_labels = test_dataloader.__iter__().next()
    #for test_images,test_labels in test_dataloader:
    
    
    batch_size = test_labels.shape[0]#N_WAY*T_QUERY_VAL
    
    
    #******************Relation-Net***************************#
    # calculate features
    v_sample_img=Variable(sample_images).cuda(GPU)
    v_test_img=Variable(test_images).cuda(GPU)
    RN_sample_features = RN_encoder(v_sample_img) # 5x64
    RN_sample_features = RN_sample_features.view(N_WAY,M_SHOT,FEATURE_DIM,4,4)
    RN_sample_features = torch.sum(RN_sample_features,1).squeeze(1)
    RN_test_features = RN_encoder(v_test_img) # 20x64
    # calculate relations
    # each batch sample link to every samples to calculate relations
    # to form a 100x128 matrix for relation network
    RN_sample_features_ext = RN_sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)
    RN_test_features_ext = RN_test_features.unsqueeze(0).repeat(1*N_WAY,1,1,1,1)
    RN_test_features_ext = torch.transpose(RN_test_features_ext,0,1)
    RN_relation_pairs = torch.cat((RN_sample_features_ext,RN_test_features_ext),2).view(-1,FEATURE_DIM*2,4,4)
    RN_certenty = relation_network(RN_relation_pairs).view(-1,N_WAY)
    #print(relations)
    #print(relations)
    #_,predict_labels = torch.max(RN_relations.data,1)
    #print('predict:',predict_labels)
    #print('reallab:',test_labels)
    
    
    #**************************Pixel-Net***************************#
    PX_support_features = PX_encoder(v_sample_img) # 5x64
            
    PX_test_features = PX_encoder(v_test_img) # 20x64
    PX_similarity = pixelsimilarity_network(PX_support_features,PX_test_features)
    PX_certentity=torch.squeeze(torch.softmax((PX_similarity-torch.mean(PX_similarity,dim=1,keepdim=True))/torch.std(PX_similarity,dim=1,keepdim=True),dim=1))
    
    
    #**************************Segmentation-Net***************************#
    
    #support_dataset=CustomDataset(df=df_support_set,class_name_dict=class_name_dict,root_dir='data/split_class',transform=transform_method)
    #query_dataset=CustomDataset(df=df_query_set,class_name_dict=class_name_dict,root_dir='data/split_class',transform=transform_method)
    SEG_support_dataset=CustomDataset(df=df_support_set,class_name_dict=class_name_dict,root_dir='data/seg',transform=transform_method)
    SEG_query_dataset=CustomDataset(df=df_query_set,class_name_dict=class_name_dict,root_dir='data/seg',transform=transform_method)
    SEG_sample_dataloader=DataLoader(dataset=SEG_support_dataset,batch_size=N_WAY*M_SHOT,shuffle=False)
    SEG_test_dataloader=DataLoader(dataset=SEG_query_dataset,batch_size=N_WAY*T_QUERY,shuffle=False)
    
    SEG_sample_images,SEG_sample_labels = SEG_sample_dataloader.__iter__().next()
    SEG_test_images,SEG_test_labels = SEG_test_dataloader.__iter__().next()
    SEG_v_sample_img=Variable(SEG_sample_images).cuda(GPU)
    SEG_v_test_img=Variable(SEG_test_images).cuda(GPU)
    SEG_support_features = PX_encoder(SEG_v_sample_img) # 5x64
            
    SEG_test_features = PX_encoder(SEG_v_test_img) # 20x64
    SEG_similarity = pixelsimilarity_network(SEG_support_features,SEG_test_features)
    SEG_certentity=torch.squeeze(torch.softmax((SEG_similarity-torch.mean(SEG_similarity,dim=1,keepdim=True))/torch.std(SEG_similarity,dim=1,keepdim=True),dim=1))

    #sample_dataloader=DataLoader(dataset=support_dataset,batch_size=N_WAY*M_SHOT,shuffle=False)
    #test_dataloader=DataLoader(dataset=query_dataset,batch_size=N_WAY*T_QUERY,shuffle=False)
        
        
        
        
        
        
    #**************************Ensemble 3 line***************************#
    
    total_cercenty=b*RN_certenty+a*PX_certentity+c*SEG_certentity
    _,predict_labels = torch.max(total_cercenty.data,1)
    rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]
    total_rewards = np.sum(rewards)
    #print(total_rewards,'in',len(rewards))
    accuracy = total_rewards/(batch_size)
    #accuracy = total_rewards/1.0/counter
    accuracies.append(accuracy)

test_accuracy,h = mean_confidence_interval(accuracies)
print("final model with a={},b={},c={} test accuracy:".format(a,b,c),test_accuracy,"h:",h)
print('this accuracy depends on the spacific task used in test, so it may be different to the acc in report')
    


