import os
import pdb
import cv2
import argparse
import random
import torch
import torchvision
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import mnist, fashion_mnist
from emnist import extract_training_samples
from data.dedb import *

# import sys
# sys.path.insert(0,'..')
from misc.utils import *
from third_party.mixture_loader.mixture import *

class DataGenerator:
    """ Data Generator
    Generating non_iid_50 task

    Created by:
        Wonyong Jeong (wyjeong@kaist.ac.kr)
    """
    def __init__(self, args):
        self.args = args
        self.seprate_ratio = (5/7, 1/7, 1/7) # train, test, valid
        self.mixture_dir = self.args.task_path
        self.mixture_filename = 'mixture.npy'
        self.base_dir = os.path.join(self.args.task_path, self.args.task)
        self.did_to_dname = {
            0: 'cifar10',
            1: 'cifar100',
            2: 'mnist',
            3: 'svhn',
            4: 'fashion_mnist',
            5: 'binary',
            6: 'face_scrub',
            7: 'not_mnist',
        }
        if self.args.task == 'non_iid_50':
            self.generate_data()
        elif self.args.task == 'mnist':
            if self.args.only_federated:
                self._generate_mnist_fed()
            else:
                self._generate_mnist()
        elif self.args.task == 'binary':
            if self.args.only_federated:
                self._generate_mnist_fed()
            else:
                self._generate_binary_task(self.args.t_name)

    def generate_data(self):
        saved_mixture_filepath = os.path.join(self.mixture_dir, self.mixture_filename)
        if os.path.exists(saved_mixture_filepath):
            print('loading mixture data: {}'.format(saved_mixture_filepath))
            mixture = np.load(saved_mixture_filepath, allow_pickle=True)
        else:
            print('downloading & processing mixture data')
            mixture = get(base_dir=self.mixture_dir, fixed_order=True)
            np_save(self.mixture_dir, self.mixture_filename, mixture)
        self.generate_tasks(mixture)

    def pad_zeros(self, arr, max_len):
        t = max_len - arr.shape[1]
        return np.pad(arr, pad_width=(0, t), mode='constant') 

    def generate_tasks(self, mixture):
        print('generating tasks ...')
        self.task_cnt = -1
        for dataset_id in self.args.datasets:
            self._generate_tasks(dataset_id, mixture[0][dataset_id])

    def _generate_tasks(self, dataset_id, data):
        # concat train & test
        x_train = data['train']['x']
        y_train = data['train']['y']
        x_test = data['test']['x']
        y_test = data['test']['y']
        x_valid = data['valid']['x']
        y_valid = data['valid']['y']
        x = np.concatenate([x_train, x_test, x_valid])
        y = np.concatenate([y_train, y_test, y_valid])

        # shuffle dataset
        idx_shuffled = np.arange(len(x))
        random_shuffle(self.args.seed, idx_shuffled)
        x = x[idx_shuffled]
        y = y[idx_shuffled]

        if self.args.task == 'non_iid_50':
            self._generate_non_iid_50(dataset_id, x, y)

    def _generate_non_iid_50(self, dataset_id, x, y):
        labels = np.unique(y)
        random_shuffle(self.args.seed, labels)
        labels_per_task = [labels[i:i+self.args.num_classes] for i in range(0, len(labels), self.args.num_classes)]
        for task_id, _labels in enumerate(labels_per_task):
            if dataset_id == 5 and task_id == 8:
                continue
            elif dataset_id in [1,6] and task_id > 15:
                continue
            self.task_cnt += 1
            idx = np.concatenate([np.where(y[:]==c)[0] for c in _labels], axis=0)
            random_shuffle(self.args.seed, idx)
            x_task = x[idx]
            y_task = y[idx]

            idx_labels = [np.where(y_task[:]==c)[0] for c in _labels]
            for i, idx_label in enumerate(idx_labels):
                y_task[idx_label] = i # reset class_id
            y_task = tf.keras.utils.to_categorical(y_task, len(_labels))

            filename = '{}_{}'.format(self.did_to_dname[dataset_id], task_id)
            self._save_task(x_task, y_task, _labels, filename, dataset_id)

    def _generate_mnist(self):
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        x_temp = np.concatenate((X_train, X_test))
        x = x_temp.reshape(x_temp.shape[0],x_temp.shape[1]*x_temp.shape[2]) #flatten 28x28 pixels to one dimension (784 inputs)
        x = np.where(x > 127, 1, 0) #binarize x
        y = np.concatenate((Y_train, Y_test))

        #shuffle dataset
        idx_shuffled = np.arange(len(x))
        random_shuffle(self.args.seed, idx_shuffled)
        x = x[idx_shuffled]
        y = y[idx_shuffled]

        # shuffle labels before label - > task mapping
        labels = np.unique(y)
        random.seed(self.args.seed)
        random.shuffle(labels)

        labels_per_task = []
        if self.args.experiment == "atten_loss":
            labels_per_task = [[8, 5, 9], [3, 0, 1], [8, 7, 6], [4, 0, 1]]
        if self.args.experiment == "attention": # Assuming 4 clients and num_tasks == 4
            #labels_per_task = [[1,7], [8,6], [9,0], [2,3], [1,7], [8,6], [9,0], [2,3]]
            labels_per_task = [[5,8,9], [0,1,3], [6,7,8], [0,1,4], [6,8,9], [4,5,6], [3,5,9], [1,4,7]]
        elif self.args.experiment == "hyperparam":
            labels_per_task = [[5, 8], [1, 4], [3, 6],
                  [0, 2], [9, 7], [5, 1],
                  [1, 6], [0, 5], [7, 2],
                  [3, 7], [2, 6], [4, 8],
                  [4, 9], [3, 8], [0, 9]]
        else:
            tasks_to_build = self.args.num_tasks * self.args.num_clients
            if tasks_to_build * self.args.num_classes <= len(labels) and len(labels) %self.args.num_classes == 0:
                labels_per_task = [labels[i:i+self.args.num_classes] for i in range(0, len(labels), _num_classes)] #range syntax: (lower bound, upper bound, step size)
            else:
                #there have to be class overlaps
                for i in range(tasks_to_build):
                    labels_per_task.append(random_sample(self.args.seed+i, labels.tolist(), self.args.num_classes))

        for task_id, _labels in enumerate(labels_per_task): #counter, value
            idx = np.concatenate([np.where(y[:]==c)[0] for c in _labels], axis=0)
            #shuffle so samples are not sampled by class cause of previous concat operation
            random.seed(self.args.seed)
            random.shuffle(idx) # shuffle order of training samples derived for current task
            x_task = x[idx]
            # y_task = y[idx] => since we are training an autoencoder,  we are not interested in class labels
            # 2 is id for mnist_dataset
            filename = '{}_{}'.format(self.did_to_dname[2], task_id)
            self._save_task(x_task, x_task, _labels, filename, 2)

        X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]) #flatten 28x28 pixels to one dimension (784 inputs)
        X_test = np.where(X_test > 127, 1, 0) #binarize x
        save_task(base_dir=self.base_dir, filename='mnist_full_test_set', data={
            'x_test' : X_test,
            'y_test' : X_test,
            'dataset_id': 2
        })

    def _generate_binary_task(self,t_name, max_len=None, g_1= False,g_2= False,g_3= False,g_4= False ):
        b_data= binary_data()
        X= {}
        labels= t_name #list(map(str, t_name.keys()))
        t_label = labels[len(labels)-1]
        t_data, _ , _ = b_data.load_debd(t_label)
        max_len= t_data.shape[1]
        print(max_len)
        #if g_1:
        for _l in labels:
            
            train,valid,test = b_data.load_debd(_l)
            len_t= train.shape[0]
            train= self.pad_zeros(train, max_len)
            train=train[:len_t]
            len_v= valid.shape[0]
            valid = self.pad_zeros(valid, max_len)
            valid= valid[:len_v]
            len_tt= test.shape[0]
            test= self.pad_zeros(test, max_len)
            test= test[:len_tt]
            temp_concat= np.concatenate((train,valid,test))
            X[_l]= np.ones((temp_concat.shape))
            X[_l] = temp_concat
                



        
        print("The Mixed Up labelling")
        tasks_to_build = self.args.num_tasks * self.args.num_clients
        #if tasks_to_build * self.args.num_classes <= len(labels) and len(labels) %self.args.num_classes == 0:
        #    labels_per_task = [labels[i:i+self.args.num_classes] for i in range(0, len(labels), _num_classes)] #range syntax: (lower bound, upper bound, step size)
        #else:
            
            #Random shuffling, mixing digits with letters for tasks, there have to be class overlaps
        labels_per_task= []
        print(labels)
        if self.args.experiment == 'other':
            for i in range(tasks_to_build):
                labels_per_task.append(random_sample(self.args.seed+i, labels, self.args.num_classes))
        elif self.args.experiment == "continual":
            labels_per_task= ['accidents', 'mushrooms', 'adult', 'connect4', 'ocr_letters', 
                                'mushrooms', 'accidents', 'ocr_letters', 'adult', 'connect4',
                                'adult',    'ocr_letters', 'connect4', 'accidents', 'mushrooms',
                                'connect4', 'adult', 'accidents', 'mushrooms', 'accidents',
                                'ocr_letters', 'connect4', 'mushrooms', 'ocr_letters', 'adult',
                                'accidents', 'mushrooms', 'adult', 'connect4', 'ocr_letters']
        elif self.args.experiment == 'label permutation ácross clients':
            temp_labels = easydict.EasyDict()
            for i in range(self.args.num_tasks):
                if i not in temp_labels:
                    temp_labels[f'{i}'] = t_name
                    #temp_labels[i].append(t_name)
            #print(temp_labels)
            for i in range(self.args.num_tasks):
                #temp_labels= {}
                temp_label = temp_labels[f'{i}']
                #print("temp labels", temp_label)
                for j in range(self.args.num_clients):
                    a= random_sample(self.args.seed+j, temp_label, self.args.num_classes)
                    temp_label.remove(a[0])
                    labels_per_task.append(a)
        elif self.args.experiment == 'label_specific':
            t_classes= []
            for _c in range(self.args.num_clients):
                a= random_sample(self.args.seed+_c, labels, self.args.num_classes)
                #print(a, labels)
                labels.remove(a[0])
                t_classes.append(a)
            for i in range(self.args.num_tasks):
                for j in range(self.args.num_clients):
                    labels_per_task.append(t_classes[j])        
        for i in range(6):   
            print(labels_per_task[5*i:5*(i+1)])

        for task_id, _labels in enumerate(labels_per_task): #counter, value
            print(task_id)
            #idx = np.concatenate([np.where(y[:]==c)[0] for c in _labels], axis=0)
            #shuffle so samples are not sampled by class cause of previous concat operation
            #print(_labels[0])
            print(len(X[_labels]))
            id_shuffled = np.arange(len(X[_labels]))
            random_shuffle(self.args.seed, id_shuffled)
            #x = x[id_shuffled]
            
            #random.seed(self.args.seed)
            #random.shuffle(idx) # shuffle order of training samples derived for current task
            x_task = X[_labels][id_shuffled]
            # y_task = y[idx] => since we are training an autoencoder,  we are not interested in class labels
            # 2 is id for mnist_dataset
            print("di_to_dname",self.did_to_dname[5])
            filename = '{}_{}'.format(self.did_to_dname[5], task_id)
            self._save_task(x_task, x_task, _labels, filename, 5)

        #for task_id in range(self.args.num_tasks * self.args.num_clients): #counter, value

        #        np.random.shuffle(X)
                #shuffle so samples are not sampled by class cause of previous concat operation
                #random.seed(self.args.seed)
                #random.shuffle(idx) # shuffle order of training samples derived for current task
        #        x_task = X
                # y_task = y[idx] => since we are training an autoencoder,  we are not interested in class labels
                # 2 is id for mnist_dataset
        #        print("di_to_dname",self.did_to_dname[5])
        #        filename = '{}_{}'.format(self.did_to_dname[5], task_id)
        #        self._save_task(x_task, x_task,labels, filename, 5)

        #for task_id in range(self.args.num_tasks * self.args.num_clients): #counter, value

        #        np.random.shuffle(X)
        #        #shuffle so samples are not sampled by class cause of previous concat operation
        #        #random.seed(self.args.seed)
        #        #random.shuffle(idx) # shuffle order of training samples derived for current task
        #        idx= math.floor(len(X) * 0.75)
        #        x_task = X[:idx]
        #        # y_task = y[idx] => since we are training an autoencoder,  we are not interested in class labels
        #        # 2 is id for mnist_dataset
        #        print("di_to_dname",self.did_to_dname[5])
        #        filename = '{}_{}'.format(self.did_to_dname[5], task_id)
        #        self._save_b_task(x_task, x_task, _labels, filename, 5, valid=valid, test=test)
        


    def _generate_mnist_fed(self):
        #(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
        data_path = self.args.mnist_path
        mnist = np.load(data_path)
        X_train, X_valid, X_test = mnist['train_data'], mnist['valid_data'], mnist['test_data']
        self._save_task(X_train, X_train, [0,1], "mnist_0", 2, X_valid, X_test)
        #x = np.concatenate((X_train, X_valid, X_test))
        #num_clients = self.args.num_clients
        #samples_per_task = len(X_train)//num_clients
        #x_temp = np.concatenate((X_train, X_valid, X_test))
        #x = x_temp.reshape(x_temp.shape[0],x_temp.shape[1]*x_temp.shape[2]) #flatten 28x28 pixels to one dimension (784 inputs)

        #shuffle dataset
        #idx_shuffled = np.arange(len(X_train))
        #random_shuffle(self.args.seed, idx_shuffled)
        #X_train = X_train[idx_shuffled]

        #for cid in range(num_clients):
        #    if cid == num_clients -1:
        #        slice = X_train[cid*samples_per_task:]
        #    else:
        #        slice = X_train[cid*samples_per_task:cid*samples_per_task+samples_per_task] #slicing stop argument is excluded
        #    labels = [0, 1] #do not matter for an autoencoder
        #    filename = '{}_{}'.format(self.did_to_dname[2], cid)
        #    self._save_task(slice, slice, labels, filename, 2, valid = X_valid, test = X_test)

        # shuffle labels before label - > task mapping
        #labels = [0, 1]

        # y_task = y[idx] => since we are training an autoencoder,  we are not interested in class labels
        # 2 is id for mnist_dataset
        #filename = '{}_{}'.format(self.did_to_dname[2], 1)
        #self._save_task(x, x, labels, filename, 2)
        save_task(base_dir=self.base_dir, filename='mnist_full_test_set', data={
            'x_test' : X_test,
            'y_test' : X_test,
            'dataset_id': 2
        })

    def _save_task(self, x_task, y_task, labels, filename, dataset_id, valid = None, test = None):
        # pairs = list(zip(x_task, y_task))
        if valid is not None:
            train_name = filename+'_train'
            save_task(base_dir=self.base_dir, filename=train_name, data={
                'x_train': x_task,
                'y_train': y_task,
                'labels': labels,
                'name': train_name,
                'dataset_id': dataset_id
            })
            valid_name = filename+'_valid'
            save_task(base_dir=self.base_dir, filename=valid_name, data={
                'x_valid': valid,
                'y_valid': valid,
                'dataset_id': dataset_id
            })
            test_name = filename+'_test'
            save_task(base_dir=self.base_dir, filename=test_name, data={
                'x_test' : test,
                'y_test' : test,
                'dataset_id': dataset_id
            })
            #print('filename:{}, labels:[{}], num_train:{}, num_valid:{}, num_test:{}'.format(filename,', '.join(map(str, labels)), len(x_task), len(valid), len(test)))
            print('filename:{}, labels:[{}], num_train:{}, num_valid:{}, num_test:{}'.format(filename,', '.join(labels, len(x_task), len(valid), len(test))))
            return
        num_examples = len(x_task)
        num_train = int(num_examples*self.seprate_ratio[0])
        num_test = int(num_examples*self.seprate_ratio[1])
        num_valid = num_examples - num_train - num_test
        train_name = filename+'_train'
        save_task(base_dir=self.base_dir, filename=train_name, data={
            'x_train': x_task[:num_train],
            'y_train': y_task[:num_train],
            'labels': labels,
            'name': train_name,
            'dataset_id': dataset_id
        })
        valid_name = filename+'_valid'
        save_task(base_dir=self.base_dir, filename=valid_name, data={
            'x_valid': x_task[num_train+num_test:],
            'y_valid': y_task[num_train+num_test:],
            'dataset_id': dataset_id
        })
        test_name = filename+'_test'
        save_task(base_dir=self.base_dir, filename=test_name, data={
            'x_test' : x_task[num_train:num_train+num_test],
            'y_test' : y_task[num_train:num_train+num_test],
            'dataset_id': dataset_id
        })

        #print('filename:{}, labels:[{}], num_train:{}, num_valid:{}, num_test:{}'.format(filename,', '.join(map(str, labels)), num_train, num_valid, num_test))
        print('filename:{}, labels:[{}], num_train:{}, num_valid:{}, num_test:{}'.format(filename,', '.join(labels), num_train, num_valid, num_test))