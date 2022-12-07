import os
import torch
import numpy as np
import hickle as hkl
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt


class UCF(Dataset):
    def __init__(self,datafile,sourcefile,nt):
        self.datafile=datafile
        self.sourcefile=sourcefile

        self.X=hkl.load(self.datafile)
        self.sources=hkl.load(self.sourcefile)
        self.nt=nt
        current_location=0
        possible_starts=[]
        while current_location<self.X.shape[0]-self.nt+1:
            if self.sources[current_location]==self.sources[current_location+self.nt-1]:
                possible_starts.append(current_location)
                current_location+=self.nt
            else:
                current_location+=1
        self.possible_starts=possible_starts

    def __getitem__(self, index):
        location = self.possible_starts[index]
        return self.preprocess(self.X[location:location + self.nt]/127.5-1)

    def __len__(self):
        return len(self.possible_starts)

    def preprocess(self, X):
        return X.astype(np.float32)





