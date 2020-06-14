import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

import os
import os.path

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))



def make_dataset(split_file, split, root, num_classes=65):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid+'.npy')):   # 特征文件
            continue
        fts = np.load(os.path.join(root, vid+'.npy'))  # (32,1024)
        num_feat = fts.shape[0]  # 32
        label = np.zeros((num_feat,num_classes), np.float32)  # (32,65)

        # 获取每帧对应的标注
        fps = num_feat/data[vid]['duration']
        for ann in data[vid]['actions']:   # 标注格式： [类别，开始时间，结束时间]
            for fr in range(0,num_feat,1):
                if fr/fps > ann[1] and fr/fps < ann[2]:   
                    label[fr, ann[0]-1] = 1 # binary classification, class index -1 to make 0 indexed
        dataset.append((vid, label, data[vid]['duration']))  # 每帧的类别标注
        i += 1
    
    return dataset

# make_dataset('multithumos.json', 'training', '/ssd2/thumos/val_i3d_rgb')

class MultiThumos(data_utl.Dataset):

    def __init__(self, split_file, split, root, batch_size):
        
        self.data = make_dataset(split_file, split, root)    # (vid, label, data[vid]['duration'])
        self.split_file = split_file
        self.batch_size = batch_size
        self.root = root
        self.in_mem = {}   # 将特征保存在字典里

    def __getitem__(self, index):
        """
        entry: (vid, label, data[vid]['duration'])
        """
        entry = self.data[index]
        if entry[0] in self.in_mem:
            feat = self.in_mem[entry[0]]
        else:
            feat = np.load(os.path.join(self.root, entry[0]+'.npy'))
            feat = feat.astype(np.float32)
            self.in_mem[entry[0]] = feat
            
        label = entry[1]
        return feat, label, [entry[0], entry[2]]  # 特征，帧级label

    def __len__(self):
        return len(self.data)


   
def mt_collate_fn(batch):
    "Pads data and puts it into a tensor of same dimensions"
    max_len = 0
    # b: (feat,label,[vid,duration])
    # 其中feat:(32,1,1,1024)  label:(32,65)
    
    # 获取feat的最长长度
    for b in batch:   # b (T,H,W,C)
        if b[0].shape[0] > max_len:
            max_len = b[0].shape[0]   #max_len = 32

    new_batch = []
    for b in batch:
        f = np.zeros((max_len, b[0].shape[1], b[0].shape[2], b[0].shape[3]), np.float32)     #(32,1,1,1024)
        m = np.zeros((max_len), np.float32)  # (32,)    
        l = np.zeros((max_len, b[1].shape[1]), np.float32)   #(32,65)
        f[:b[0].shape[0]] = b[0]
        m[:b[0].shape[0]] = 1   # [1,1,1,.....1]
        l[:b[0].shape[0], :] = b[1]
        new_batch.append([video_to_tensor(f), torch.from_numpy(m), torch.from_numpy(l), b[2]])  # (T,H,W,C)---->(C,T,H,W)

    # 注意输出特征的格式为(B,C,T,H,W)
    return default_collate(new_batch)
    
