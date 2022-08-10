from cv2 import dct
import torch
from torch.utils.data import Dataset,DataLoader
import os
import pickle
import cv2
import numpy as np

class Ind_dataset(Dataset):
    def __init__(self,dir_path,transform=None, target_transfrom = None, LabelDic = {'YT':0,'WA':1,'org':2}):
        dtype = os.path.split(dir_path)[1]
        self.dir_path = dir_path
        self.pickle_path = []
        with open(os.path.join(dir_path,dtype+'.txt'),'r') as f:
            for i in f.readlines():
                self.pickle_path.append(i.strip())
        self.transform = transform
        self.target_transform = target_transfrom
        self.LabelDic = LabelDic
        self.dir_path = dir_path

    def __len__(self):
        return len(self.pickle_path)

    def __getitem__(self, index):
        path = self.pickle_path[index].split(' ')[0].strip()
        label = self.pickle_path[index].split(' ')[-1].strip()
        path = os.path.join(self.dir_path,path)
        with open(path,'rb') as f:
            img = pickle.load(f)
        img = self.transfrom_img(img)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        if self.LabelDic != None:
            label = self.LabelDic[label]
        return img,label
    
    def get_label(self):
        label = []
        for item in self.pickle_path:
            label.append(self.LabelDic[item.split(' ')[-1].strip()])
        return label

    def transfrom_img(self, img):
        cv_img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        ycbcr_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2YCR_CB)
        y_channel = ycbcr_img[:,:,0]
        Gaussian_y = cv2.GaussianBlur(y_channel,(11,11),0) 
        hpf_yimg = y_channel-Gaussian_y
        return hpf_yimg
    

class Pred_dataset(Dataset):
    def __init__(self,dir_path,mini_set = True,transform=None, target_transfrom = None, LabelDic = {'YT':0,'WA':1,'org':2}):
        print(dir_path)
        dtype = os.path.split(dir_path)[1]
        self.dir_path = dir_path
        self.pickle_path = []
        self.mini_set = mini_set       
        file_path = os.path.join(dir_path,dtype+'.txt')
        with open(file_path,'r') as f:
            for i in f.readlines():
                self.pickle_path.append(i.strip())
        self.transform = transform
        self.target_transform = target_transfrom
        self.LabelDic = LabelDic
        self.dir_path = dir_path

    def __len__(self):
        return len(self.pickle_path)

    def __getitem__(self, index):
        img = []
        for i in range(1,4):
            path = self.pickle_path[index].split(' ')[i].strip()
            path = os.path.join(self.dir_path,path)
            with open(path,'rb') as f:
                img.append(pickle.load(f))
        img = np.array(img)
        label = self.pickle_path[index].split(' ')[-1].strip()
        img = self.transfrom_img(img)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        if self.LabelDic != None:
            label = self.LabelDic[label]
        return img,label

    def transfrom_img(self,img):
        P_patch = []
        for patch in img:
            cv_img = cv2.cvtColor(patch,cv2.COLOR_RGB2BGR)
            ycbcr_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2YCR_CB)
            y_channel = ycbcr_img[:,:,0]
            Gaussian_y = cv2.GaussianBlur(y_channel,(11,11),0) 
            residue = y_channel-Gaussian_y
            P_patch.append(residue)
        return np.array(P_patch).transpose((1,2,0)) 

class Multi_dataset(Dataset):
    def __init__(self,dir_path,transform=None, target_transfrom = None, LabelDic = {'YT':0,'WA':1,'org':2}):
        self.dir_path = dir_path
        self.transfrom = transform
        self.target_transfrom = target_transfrom
        self.LabelDic = LabelDic
        dtype = os.path.split(dir_path)[1]
        self.pickle_path = []
        with open(os.path.join(dir_path,dtype+'.txt'),'r') as f:
            for i in f.readlines():
                self.pickle_path.append(i.strip())
    
    def __len__(self):
        return len(self.pickle_path)
    
    def __getitem__(self, index):
        I_path = self.pickle_path[index].split(' ')[0].strip()
        with open(os.path.join(self.dir_path,I_path),'rb') as f:
            I_patch = pickle.load(f)
        P_patch = []
        for i in range(1,4):
            path = self.pickle_path[index].split(' ')[i].strip()
            path = os.path.join(self.dir_path,path)
            with open(path,'rb') as f:
                P_patch.append(pickle.load(f))
        label = self.pickle_path[index].split(' ')[-1].strip()
        I_patch = np.array(I_patch)
        P_patch = np.array(P_patch)
        I_patch = self.transfrom_I(I_patch)
        P_patch = self.transfrom_P(P_patch)
        if self.transfrom:
            I_patch = self.transfrom(I_patch)
            P_patch = self.transfrom(P_patch)
        if self.target_transfrom:
            label = self.target_transfrom(label)
        if self.LabelDic:
            label = self.LabelDic[label]
        return [I_patch,P_patch],label
            

    def transfrom_I(self,img):
        cv_img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        ycbcr_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2YCR_CB)
        y_channel = ycbcr_img[:,:,0]
        Gaussian_y = cv2.GaussianBlur(y_channel,(11,11),0) 
        hpf_yimg = y_channel-Gaussian_y
        return hpf_yimg

    def transfrom_P(self,img):
        P_patch = []
        for patch in img:
            cv_img = cv2.cvtColor(patch,cv2.COLOR_RGB2BGR)
            ycbcr_img = cv2.cvtColor(cv_img,cv2.COLOR_BGR2YCR_CB)
            y_channel = ycbcr_img[:,:,0]
            Gaussian_y = cv2.GaussianBlur(y_channel,(11,11),0) 
            residue = y_channel-Gaussian_y
            P_patch.append(residue)
        return np.array(P_patch).transpose((1,2,0)) 

