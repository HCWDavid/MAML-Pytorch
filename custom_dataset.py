from collections import defaultdict
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import random
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import CubicSpline
import tqdm

def ctr_seed(sd=0, cudnn=False, deterministic=False):
    np.random.seed(sd)
    random.seed(sd)
    torch.manual_seed(sd)
    torch.backends.cudnn.benchmark = cudnn
    torch.cuda.manual_seed(sd)
    torch.use_deterministic_algorithms(deterministic)


# from vector_magnitude import calculate_magnitude
def adjust_labels(y, left_size, right_size):
    # Left side label, increasing from 0 to 1
    # left_labels = torch.linspace(0, 1, steps=left_size)
    left_labels = torch.zeros(left_size)

    # Right side label, decreasing from 1 to 0
    # right_labels = torch.linspace(1, 0, steps=right_size)
    right_labels = torch.zeros(right_size)

    return torch.cat([left_labels, y, right_labels], dim=0)

def GenerateRandomCurves(X, sigma=0.1, knot=4):
    # https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb
    # print("Shape of X:", X.shape)
    xx = (np.ones((X.shape[1], 1)) * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    res = [CubicSpline(xx[:, i], yy[:, i])(x_range) for i in range(len(X.T))]
    return X * np.array(res).transpose()


def add_length_both_sides(x, left_mean, right_mean):
    dim1 = left_mean.shape[0]
    left_size = 150
    right_size = 150
    left_new_data = left_mean + torch.randn(size=(left_size, dim1)) * .05
    right_new_data = right_mean + torch.randn(size=(right_size, dim1)) * .05
    return torch.cat([left_new_data, x, right_new_data], dim=0)

def soft_label_generator(length):
    half_n = length // 2
    x1 = torch.linspace(0, 3.141592653589793, half_n)
    y1 = (1 - torch.cos(x1)) / 2  # Goes from 0 to 1

    # Create linear space from pi to 2*pi for the second half
    x2 = torch.linspace(3.141592653589793, 2 * 3.141592653589793, length - half_n)
    y2 = (1 + torch.cos(x2)) / 2  # Goes from 1 to 0
    y2 = torch.flip(y2, dims=(0,))
    # Concatenate the two halves to form a full array
    return torch.cat([y1, y2])

class sliding_windows(nn.Module):
    def __init__(self, total_length, width, step):
        # https://stackoverflow.com/questions/53972159/how-does-pytorchs-fold-and-unfold-work
        super(sliding_windows, self).__init__()
        self.total_length = total_length
        self.width = width
        self.step = step

    def forward(self, input_time_series, labels):
        
        input_transformed = torch.swapaxes(input_time_series.unfold(-2, size=self.width, step=self.step), -2, -1)
        # For labels, we only have one dimension, so we unfold along that dimension
        labels_transformed = labels.unfold(0, self.width, self.step)
        return input_transformed, labels_transformed

    def get_num_sliding_windows(self):
        return round((self.total_length - (self.width - self.step)) / self.step)

# NOTE: sec implementation of Custom Dataset that works like nn.Dataset, .e.g.: MNIST(train=True...):

def fold(x: torch.Tensor, step: int, width: int) -> torch.Tensor:
    # Asserting x is either 2D or 3D
    assert 2 <= x.dim() <= 3, "Input tensor must be either 2D or 3D."

    # Calculating total length
    N = x.size(0)
    total_length = (N - 1) * step + width

    # Initialize an empty tensor to store the folded result
    if x.dim() == 2:
        folded = torch.zeros(total_length, device=x.device)
    else:
        folded = torch.zeros(total_length, x.size(1), device=x.device)
    
    # Folding operation
    for i in range(N):
        start_idx = i * step
        end_idx = start_idx + width
        if x.dim() > 2:
            folded[start_idx:end_idx] += torch.permute(x[i], (1,0))
        else:
            folded[start_idx:end_idx] += x[i]
    if x.dim() > 2:
        folded = folded.permute(1, 0)

    return folded

class unsliding_windows(nn.Module):
    def __init__(self, width, step):
        super(unsliding_windows, self).__init__()
        self.width = width
        self.step = step

    def forward(self, input_time_series):
        reconstructed = fold(input_time_series, self.step, self.width)
        return reconstructed



class PHYSIQDataset(Dataset):
    def __init__(self, train=True, 
                 shuffle=True, 
                 y_label='segmentation', 
                 transform=None, 
                 option='soft', 
                 slide_windows=True, 
                 window_size=200, 
                 window_step=200, 
                 min_rep=5, 
                 rep_step=1, 
                 seed=12345):
        super(PHYSIQDataset, self).__init__()
        ctr_seed(sd=seed)
        self.train = train
        self.transform = transform
        self.option = option
        self.shuffle = shuffle
        self.slide_windows = slide_windows
        self.window_size = window_size
        self.window_step = window_step
        self.min_rep = min_rep
        self.rep_step = rep_step
        self.data = None
        self.y_label = y_label
        self.targets = None
        self._load_dataset()

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.x[index]), self.y[index]
        return self.x[index], self.y[index]
    def get_dataset_params(self):
        # return as a dict:
        return {
            'option': self.option,
            'slide_windows': self.slide_windows,
            'window_size': self.window_size,
            'window_step': self.window_step,
            'min_rep': self.min_rep,
            'rep_step': self.rep_step
        }
    def check_params_the_same(self, dict2):
        # check if the dataset params are the same:
        dict1 = self.get_dataset_params()
        for i in dict1:
            if dict1.get(i)!=dict2.get(i):
                return False
        return True
    
    def get_x_y(self, data):
        self.x = data['x']
        if self.y_label == "segmentation":
            self.y = data['segmentation']
        elif self.y_label == "exercise_label":
            self.y = data['exercise_label']
        elif self.y_label == "subject":
            self.y = data['subject']
        elif self.y_label == "num_rep":
            if self.slide_windows:
                raise AssertionError("num_rep is not supported when slide_windows is True, because no num_rep is available after x is slided.")
            self.y = data['num_rep']
        elif self.y_label == "all":
            # return all as a tuple of (seg, exer, subj, num_rep)
            if self.slide_windows:
                self.y = zip(data['segmentation'], data['exercise_label'], data['subject'])
            else:
                self.y = zip(data['segmentation'], data['exercise_label'], data['subject'], data['num_rep'])
            self.y = list(self.y)
        else:
            raise AssertionError(f"y_label: {self.y_label} is not supported.")
        # train-valid split:
        # Shuffle x and y together:
        if self.shuffle:
            perm = torch.randperm(self.x.size(0))
        else:
            perm = torch.arange(self.x.size(0))
        x_shuffled = self.x[perm]
        # self.y could be a list:
        if type(self.y) == list:
            y_shuffled = [self.y[i] for i in perm]
        else:
            y_shuffled = self.y[perm]
        length = int(len(x_shuffled)*0.6875) # 11 subject for train, 5 for valid for PHYSIQ
        if self.train == True:
            self.x = x_shuffled[:length]
            self.y = y_shuffled[:length]
        elif self.train == False:
            self.x = x_shuffled[length:]
            self.y = y_shuffled[length:]
        elif self.train == None:
            self.x = x_shuffled
            self.y = y_shuffled
        else:
            raise AssertionError(f"train: {self.train} is not supported.")
        return self.x, self.y

    
    def _load_dataset(self):
        if self.slide_windows:
            pt_filename = './dataset_pickle/PHYSIQ_ss.pt'
        else:
            pt_filename = './dataset_pickle/PHYSIQ_ns.pt'
        # check if it exists:
        if not os.path.exists(pt_filename):
            self._load_pickle_data()
        else:
            data = torch.load(pt_filename)
            # a dictionary with keys: ['x', 'segmentation', 'exercise_label', 'subject', 'num_rep']
            # print(data)
            if self.check_params_the_same(data):
                self.get_x_y(data)
            else:
                self._load_pickle_data()
        return
    

    def _load_pickle_data(self):
        pickle_filename = './dataset_pickle/PHYSIQ.pickle'
        # check if it exists:
        if not os.path.exists(pickle_filename):
            raise AssertionError(f"{pickle_filename} does not exist. Need to find way to download or create it.")
        with open(pickle_filename, 'rb') as f:
            data = pickle.load(f)
        # print(data.keys())
        prior_res = self._create_label(data) 
        if self.slide_windows:
            res = self._slide_and_augment(prior_res)
            res.update(self.get_dataset_params())
            torch.save(res, './dataset_pickle/PHYSIQ_ss.pt')
        else:
            res = prior_res
            res.update(self.get_dataset_params())
            torch.save(res, './dataset_pickle/PHYSIQ_ns.pt')
        self.get_x_y(res)
        return


        # get the data and process it:

    
    def _slide_and_augment(self, dict):
        
        data = dict['x']
        labels = dict['segmentation']
        task = dict['exercise_label']
        subject = dict['subject']

        output_data = []
        output_labels = []
        output_tasks = []
        output_subjects = []

        for i in tqdm.tqdm(range(len(data))):
            data[i] = torch.from_numpy(GenerateRandomCurves(data[i].numpy()))
            total_length = data[i].shape[0]  # Calculate the length of the current time series
            self.sliding_window_transform = sliding_windows(total_length, self.window_size, self.window_step)

            # Apply sliding window transformation to x and y
            temp_data, temp_label = self.sliding_window_transform(data[i], labels[i])
            output_data.append(temp_data)
            output_labels.append(temp_label)
            output_tasks.append(torch.full((temp_data.shape[0],), task[i]))
            output_subjects.append(torch.full((temp_data.shape[0],), subject[i]))
        
        # change the list to tensor
        output_data = torch.cat(output_data, dim=0)
        output_labels = torch.cat(output_labels, dim=0)
        output_tasks = torch.cat(output_tasks, dim=0)
        output_subjects = torch.cat(output_subjects, dim=0)
        return {
            'x': output_data,
            'segmentation': output_labels,
            'exercise_label': output_tasks,
            'subject': output_subjects,
            'num_rep': None
        }
        # return output_data, output_labels, output_tasks

    
    def _load_data(self, data):
        inputs = dict()
        subjects = []
        repetitions = []  # To store exercise numbers
        exercise_nums = [] # To store subject numbers
        subj_to_exercise = defaultdict(list)
        subj_cls_to_rep = defaultdict(list)
        for each_x, each_s, each_rep, exercise_num in zip(data['X'], data['subject'], data['repetition'], data['exercise']):
            inputs[(each_s, each_rep, exercise_num)] = each_x
            subjects.append(each_s)
            repetitions.append(each_rep)
            exercise_nums.append(exercise_num)
            subj_cls_to_rep[(each_s, exercise_num)].append(each_rep)
            subj_to_exercise[each_s].append(exercise_num)
        # print(inputs.keys())
        subj_cls_to_nums_rep = dict()
        for l, v in subj_cls_to_rep.items():
            subj_cls_to_nums_rep[l] = len(v)
        return inputs, subjects, repetitions, exercise_nums, subj_cls_to_nums_rep, subj_to_exercise

    def _generate_one_sample_segmentation_label(self, each_x):
        assert self.option in ['hard', 'soft']
        assert len(each_x.shape) == 2
        label = torch.zeros(each_x.shape[0])
        if self.option == 'hard':
            label[0] = 1
            label[-1] = 1
        elif self.option == 'soft':
            label = soft_label_generator(each_x.shape[0])
        return label
    
    def _convert_subject_to_int(self, subjects):
        """
        subjects: a list of subjects or a single subject
                i.e.: ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
                return: [1, 2, 3, 4, 5, 6, 7]

                OR 
                i.e.: 'S1'
                return: 1
        """
        if type(subjects) == list:
            return [int(each_s[1::]) for each_s in subjects]
        else:
            return int(subjects[1::])
    def _create_label(self, data):
        res_inputs = []
        res_labels = []
        res_tasks = []  # To store exercise numbers
        res_subjects = [] # To store subject numbers
        res_num_reps = []
        
        inputs, subjects, repetitions, exercise_cls, subj_cls_to_nums_rep, subj_to_exercise = self._load_data(data)

        # for sub_seq_length in range(min_rep, total_rep+1, rep_step):
        #     for index in range(max_repetition):
        #         end_index = index + sub_seq_length - 1 
        #         if end_index > max_repetition:
        #             break
        #         id_list.append([i for i in range(index, end_index + 1)])
        for subject, exercise_nums in tqdm.tqdm(subj_to_exercise.items()):
            for exercise_num in exercise_nums:
                total_rep = subj_cls_to_nums_rep[(subject, exercise_num)] # total number of repetitions
                for sub_seq_length in range(self.min_rep, total_rep, self.rep_step):
                    
                    for index in range(total_rep):
                        end_index = index + sub_seq_length
                        if end_index > total_rep:
                            # can't go beyond the total number of repetitions
                            break
                        temp_np = np.empty((0, 6)) # create one sample
                        temp_label = np.empty(0)
                        # print(end_index)
                        for each_id in range(index, end_index):
                            # print(inputs[(subject, each_id, exercise_num)])
                            temp_np = np.concatenate((temp_np, inputs[(subject, each_id, exercise_num)]), axis=0)
                            temp_label = np.concatenate((temp_label, self._generate_one_sample_segmentation_label(inputs[(subject, each_id, exercise_num)])))
                        
                        # saving:
                        res_inputs.append(torch.tensor(temp_np))
                        res_labels.append(torch.tensor(temp_label)) # create one sample segmentation label
                        res_tasks.append(exercise_num)
                        # print(self._convert_subject_to_int(subject))
                        res_subjects.append(self._convert_subject_to_int(subject))
                        res_num_reps.append(sub_seq_length)
                        

                    if (subject, each_id, exercise_num) not in inputs:
                        raise IndexError(f"({subject}, {each_id}, {exercise_num}) not in inputs.")
                        # print(temp_np.shape)

        print(f"total of inputs: {len(inputs)}")
        # print(f"total of labels: {len(labels)}")
        return {'x': res_inputs, # a list of tensors
                'segmentation': res_labels,  # a list of tensors that has the same length as x => (N, Time, 6) -> (N, Time, 1)
                'exercise_label': torch.tensor(res_tasks), 
                'subject': torch.tensor(res_subjects), 
                'num_rep': torch.tensor(res_num_reps)}



#NOTE: first implementation of Custom Dataset that works (bad implementation): ----------------------------------------------------------------------------------------

# input 500*6 ->(b*n)*200*6
class CustomDataset(Dataset):
    def __init__(self, pickle_filename='./dataset_pickle/PHYSIQ.pickle', option='soft', slide_windows=True, window_size=200, window_step=200, vm=False, random_inverse=False, min_rep=5, rep_step=1, save_raw=False):
        super(CustomDataset, self).__init__()
        ctr_seed(sd=12345)
        # Read the CSV file
        with open(pickle_filename, 'rb') as f:
            data = pickle.load(f)
        print(data.keys())
        self.random_inverse =random_inverse
        self.option = option
        self.window_size = window_size
        self.window_step = window_step
        self.min_rep = min_rep
        self.rep_step = rep_step
        self.keys = list(data.keys()) # ['X', 'subject', 'exercise', 'repetition']
        # print(data['repetition']) # data['subject'], data['repetition'])
        
        #
        prior_slide_dataset = self._create_label(data)
        if save_raw:
            self.prior_slide_dataset = prior_slide_dataset
        if slide_windows:
            self.post_slide_data = self._slide_and_augment(prior_slide_dataset)
            # self.x = self.post_slide_data['x']
            # self.y = self.post_slide_data['segmentation']
            # self.task = self.post_slide_data['exercise_label']

    def _slide_and_augment(self, dict):
        
        data = dict['x']
        labels = dict['segmentation']
        task = dict['exercise_label']
        subject = dict['subject']

        # output_data = torch.empty(0, dtype=torch.float32)
        # output_labels = torch.empty(0, dtype=torch.float32)
        # output_tasks = torch.empty(0, dtype=torch.float32)
        # output_subjects = torch.empty(0, dtype=torch.float32)
        output_data = []
        output_labels = []
        output_tasks = []
        output_subjects = []

        for i in tqdm.tqdm(range(len(data))):
            data[i] = torch.from_numpy(GenerateRandomCurves(data[i].numpy()))
            total_length = data[i].shape[0]  # Calculate the length of the current time series
            self.sliding_window_transform = sliding_windows(total_length, self.window_size, self.window_step)

            # Apply sliding window transformation to x and y
            temp_data, temp_label = self.sliding_window_transform(data[i], labels[i])
            # https://stackoverflow.com/questions/24935984/python-code-become-slower-after-each-iteration
            # no more use of torch.cat for arbitrary array as it copy the whole array and detoriate the performance:
            # output_data = torch.cat([output_data, temp_data], dim=0)
            # output_labels = torch.cat([output_labels, temp_label], dim=0)
            # output_tasks = torch.cat([output_tasks, torch.full((temp_data.shape[0],), task[i])], dim=0)
            # output_subjects = torch.cat([output_subjects, torch.full((temp_data.shape[0],), subject[i])], dim=0)
            # instead doing:
            output_data.append(temp_data)
            output_labels.append(temp_label)
            output_tasks.append(torch.full((temp_data.shape[0],), task[i]))
            output_subjects.append(torch.full((temp_data.shape[0],), subject[i]))
        
        # change the list to tensor
        output_data = torch.cat(output_data, dim=0)
        output_labels = torch.cat(output_labels, dim=0)
        output_tasks = torch.cat(output_tasks, dim=0)
        output_subjects = torch.cat(output_subjects, dim=0)

        return {
            'x': output_data,
            'segmentation': output_labels,
            'exercise_label': output_tasks,
            'subject': output_subjects,
            'num_rep': None
        }
        # return output_data, output_labels, output_tasks



    def _slide_data_labels(self, data, labels, window_size=50, step_size=10, left_size=150, right_size=150):
        raise AssertionError # This function is not used anymore
        output_data = torch.empty(0, dtype=torch.float32)
        output_labels = torch.empty(0, dtype=torch.float32)
        
        for i in range(len(data)):
            if data[i].nelement() > 0:  # Check if the tensor is non-empty
                data[i] = torch.from_numpy(GenerateRandomCurves(data[i].numpy()))

                # calculate the means for each individual sequence
                n = min(5, data[i].shape[0])  # take the minimum to avoid index error when data[i] is small

                left_mean = data[i][:n].mean(dim=0)
                right_mean = data[i][-n:].mean(dim=0)

                assert left_mean.shape == (6,) or left_mean.shape == (2,)
                assert right_mean.shape == (6,) or right_mean.shape == (2,)

                # add noise on both sides
                data[i] = add_length_both_sides(data[i], left_mean, right_mean)
                labels[i] = adjust_labels(labels[i], left_size, right_size)

                # initialize sliding window
                total_length = data[i].shape[0]  # Calculate the length of the current time series
                self.sliding_window_transform = sliding_windows(total_length, window_size, step_size)

                # Apply sliding window transformation to x and y
                temp_data, temp_label = self.sliding_window_transform(data[i], labels[i])
                output_data = torch.cat([output_data, temp_data], dim=0)
                output_labels = torch.cat([output_labels, temp_label], dim=0)

            else:
                print(f"data[{i}] is empty.")
        print(f"Shape of output_data: {output_data.shape}")
        return output_data, output_labels

    def _load_data(self, data):
        inputs = dict()
        subjects = []
        repetitions = []  # To store exercise numbers
        exercise_nums = [] # To store subject numbers
        subj_to_exercise = defaultdict(list)
        subj_cls_to_rep = defaultdict(list)
        for each_x, each_s, each_rep, exercise_num in zip(data['X'], data['subject'], data['repetition'], data['exercise']):
            inputs[(each_s, each_rep, exercise_num)] = each_x
            subjects.append(each_s)
            repetitions.append(each_rep)
            exercise_nums.append(exercise_num)
            subj_cls_to_rep[(each_s, exercise_num)].append(each_rep)
            subj_to_exercise[each_s].append(exercise_num)
        print(inputs.keys())
        subj_cls_to_nums_rep = dict()
        for l, v in subj_cls_to_rep.items():
            subj_cls_to_nums_rep[l] = len(v)
        return inputs, subjects, repetitions, exercise_nums, subj_cls_to_nums_rep, subj_to_exercise

    def _generate_one_sample_segmentation_label(self, each_x):
        assert self.option in ['hard', 'soft']
        assert len(each_x.shape) == 2
        label = torch.zeros(each_x.shape[0])
        if self.option == 'hard':
            label[0] = 1
            label[-1] = 1
        elif self.option == 'soft':
            label = soft_label_generator(each_x.shape[0])
        return label
    
    def _convert_subject_to_int(self, subjects):
        """
        subjects: a list of subjects or a single subject
                i.e.: ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']
                return: [1, 2, 3, 4, 5, 6, 7]

                OR 
                i.e.: 'S1'
                return: 1
        """
        if type(subjects) == list:
            return [int(each_s[1::]) for each_s in subjects]
        else:
            return int(subjects[1::])
    def _create_label(self, data):
        res_inputs = []
        res_labels = []
        res_tasks = []  # To store exercise numbers
        res_subjects = [] # To store subject numbers
        res_num_reps = []
        
        inputs, subjects, repetitions, exercise_cls, subj_cls_to_nums_rep, subj_to_exercise = self._load_data(data)

        # for sub_seq_length in range(min_rep, total_rep+1, rep_step):
        #     for index in range(max_repetition):
        #         end_index = index + sub_seq_length - 1 
        #         if end_index > max_repetition:
        #             break
        #         id_list.append([i for i in range(index, end_index + 1)])
        for subject, exercise_nums in tqdm.tqdm(subj_to_exercise.items()):
            for exercise_num in exercise_nums:
                total_rep = subj_cls_to_nums_rep[(subject, exercise_num)] # total number of repetitions
                for sub_seq_length in range(self.min_rep, total_rep, self.rep_step):
                    
                    for index in range(total_rep):
                        end_index = index + sub_seq_length
                        if end_index > total_rep:
                            # can't go beyond the total number of repetitions
                            break
                        temp_np = np.empty((0, 6)) # create one sample
                        temp_label = np.empty(0)
                        # print(end_index)
                        for each_id in range(index, end_index):
                            # print(inputs[(subject, each_id, exercise_num)])
                            temp_np = np.concatenate((temp_np, inputs[(subject, each_id, exercise_num)]), axis=0)
                            temp_label = np.concatenate((temp_label, self._generate_one_sample_segmentation_label(inputs[(subject, each_id, exercise_num)])))
                        
                        # saving:
                        res_inputs.append(torch.tensor(temp_np))
                        res_labels.append(torch.tensor(temp_label)) # create one sample segmentation label
                        res_tasks.append(exercise_num)
                        # print(self._convert_subject_to_int(subject))
                        res_subjects.append(self._convert_subject_to_int(subject))
                        res_num_reps.append(sub_seq_length)
                        

                    if (subject, each_id, exercise_num) not in inputs:
                        raise IndexError(f"({subject}, {each_id}, {exercise_num}) not in inputs.")
                        # print(temp_np.shape)

        # for each_x, each_s, each_rep, exercise_num in zip(inputs, subjects, repetitions, exercise_cls):
        #     a_new_x_i = torch.tensor([])
        #     a_new_y_i = torch.tensor([]) # segmentation_labels
        #     each_s = int(each_s[1::])
        #     each_rep = int(each_rep)
        #     if each_s != pri_subject:

        print(f"total of inputs: {len(inputs)}")
        # print(f"total of labels: {len(labels)}")
        return {'x': res_inputs, # a list of tensors
                'segmentation': res_labels,  # a list of tensors that has the same length as x => (N, Time, 6) -> (N, Time, 1)
                'exercise_label': torch.tensor(res_tasks), 
                'subject': torch.tensor(res_subjects), 
                'num_rep': torch.tensor(res_num_reps)}




    def __len__(self):
        # Returns the size of the dataframe
        return len(self.post_slide_data['x'])

    def __getitem__(self, idx):
        # Return the data and label at the specified index
        features = self.post_slide_data['x'][idx]
        label = self.post_slide_data['segmentation'][idx]
            
        return features, label
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj


if __name__ == "__main__":
    # cd = CustomDataset()
    # save_object(cd, './dataset_pickle/CustomDatasetPhysiQ[5-10rep].pkl')
    # cd = load_object('./dataset_pickle/CustomDatasetPhysiQ[5-10rep].pkl')
    # print(cd.post_slide_data['x'].shape, cd.post_slide_data['segmentation'].shape, cd.post_slide_data['exercise_label'].shape)

    # val_dataset = CustomDataset(pickle_filename='./dataset_pickle/SPAR.pickle', 
    #                         window_size=200)

    cd = PHYSIQDataset(y_label='all', slide_windows=True, window_size=200, window_step=200, min_rep=5, rep_step=1)
    print(cd.x.shape, len(cd.y))

    print(cd[0][0].shape, cd[0][1])