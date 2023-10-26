import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random

import tqdm

from custom_dataset import PHYSIQDataset


class TimeSeriesDataset(Dataset):

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query,resize=None, T=100, startidx=0):
        """
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """
        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.setsz = self.n_way * self.k_shot  # num of samples per set
        self.querysz = self.n_way * self.k_query  # number of samples per set for evaluation
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query' % (
        mode, batchsz, n_way, k_shot, k_query))
        
        
        #TODO: should be ts2label but using img2label for now
        self.data, self.img2label, self.ts2segms, self.ts = self.loadPHYSIQ()
        self.cls_num = len(self.data)
        print('class num:', self.cls_num, [len(v) for v in self.data])
        self.transform = transforms.Compose([lambda i: self.ts[i],])
                                                 
        
        
        self.create_batch(self.batchsz)


    def loadPHYSIQ(self, type='physiq'):
        if type == 'physiq':
            cd = PHYSIQDataset(y_label='all', train=None, slide_windows=True, window_size=200, window_step=200, min_rep=5, rep_step=1)
        else:
            raise AssertionError('Dataset type not supported')
        data = [] # [[ts1, ts2, ...], [ts111, ...]]
        ts2label = {}
        ts2segms = {} # len(segmentations) = len(cd.x)
        ts = []
        for i, (x, y) in enumerate(zip(cd.x, cd.y)):
            segm, task, subject = y

            if int(task) >= len(data):
                for i in range(int(task) - len(data) + 1):
                    data.append([])
            
            
            
            data[int(task)].append(i)
            ts2segms[i] = segm
            ts2label[i] = int(task)
            ts.append(x)
        return data, ts2label, ts2segms, ts



    def create_batch(self, batchsz):
        """
        create batch for meta-learning.
        ×episode× here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """
        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        for b in tqdm.tqdm(range(batchsz)):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(self.cls_num, self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            support_x = []
            query_x = []
            for cls in selected_cls:
                # 2. select k_shot + k_query for each class
                selected_imgs_idx = np.random.choice(len(self.data[cls]), self.k_shot + self.k_query, False) 
                np.random.shuffle(selected_imgs_idx)
                indexDtrain = np.array(selected_imgs_idx[:self.k_shot])  # idx for Dtrain
                indexDtest = np.array(selected_imgs_idx[self.k_shot:])  # idx for Dtest
                support_x.append(
                    np.array(self.data[cls])[indexDtrain].tolist())  # get all images filename for current Dtrain
                query_x.append(np.array(self.data[cls])[indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)
            random.shuffle(query_x)
            self.support_x_batch.append(support_x)  # append set to current sets
            self.query_x_batch.append(query_x)  # append sets to current sets

    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # time series has no resize
        # [setsz, 3, resize, resize]
        support_x = torch.FloatTensor(self.setsz, 200, 6)
        # [setsz]
        support_y = np.zeros((self.setsz), dtype=np.intc)
        # [querysz, 3, resize, resize]
        query_x = torch.FloatTensor(self.querysz, 200, 6)
        # [querysz]
        query_y = np.zeros((self.querysz), dtype=np.intc)
        # making it flatten by appending 'item' which are just idex of the list of time series
        flatten_support_x = [item
                             for sublist in self.support_x_batch[index] for item in sublist]
        
        support_y = np.array(
            [self.img2label[item]  # filename:n0153282900000005.jpg, the first 9 characters treated as label
             for sublist in self.support_x_batch[index] for item in sublist]).astype(np.int32)

        flatten_query_x = [item
                           for sublist in self.query_x_batch[index] for item in sublist]
        query_y = np.array([self.img2label[item]
                            for sublist in self.query_x_batch[index] for item in sublist]).astype(np.int32)
        
        # print('global:', support_y, query_y)
        # support_y: [setsz]
        # query_y: [querysz]
        # unique: [n-way], sorted
        unique = np.unique(support_y)
        random.shuffle(unique)
        # relative means the label ranges from 0 to n-way
        support_y_relative = np.zeros(self.setsz)
        query_y_relative = np.zeros(self.querysz)
        for idx, l in enumerate(unique):
            support_y_relative[support_y == l] = idx
            query_y_relative[query_y == l] = idx

        # print('relative:', support_y_relative, query_y_relative)

        for i, path in enumerate(flatten_support_x):
            support_x[i] = self.transform(path)
        for i, path in enumerate(flatten_query_x):
            query_x[i] = self.transform(path)
        # print(support_set_y)
        # return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

        return support_x, torch.LongTensor(support_y_relative), query_x, torch.LongTensor(query_y_relative)

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.batchsz

def visualize_time_series(ts, ts_y=None):

    fig, axes = plt.subplots(ts.shape[0], 1, figsize=(10, 12), sharex=True)

    # Iterate over time series
    for i in range(5):
        # Iterate over channels within a time series
        for j in range(6):
            axes[i].plot(ts[i, :, j], label=f"Channel {j+1}")
        axes[i].set_title(f"Time Series {i+1 }" + f'and y={ts_y[i]}' if ts_y is not None else "")
        axes[i].legend()
        
    return fig

if __name__ == '__main__':
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time

    tb = SummaryWriter(log_dir="runs/ts", comment='physiq')
    #TODO: mode of train and val and test does not work yet and will make sure work for custom dataset of physiq
    mini = TimeSeriesDataset('./', mode='train', n_way=5, k_shot=1, k_query=15, batchsz=1, resize=168)

    for i, set_ in enumerate(mini):
        support_x, support_y, query_x, query_y = set_
        print(query_x.shape)
        fig = visualize_time_series(support_x.numpy(), support_y.numpy())
        plt.pause(5)
        tb.add_figure('support_x', fig, global_step=i)  # Added step
        plt.close(fig)  # Close figure after adding it to tensorboard

        fig = visualize_time_series(query_x.numpy(), query_y.numpy())
        plt.pause(5)
        tb.add_figure('query_x', fig, global_step=i)  # Added step
        plt.close(fig)  # Close figure after adding it to tensorboard

        time.sleep(5)

    tb.close()
    