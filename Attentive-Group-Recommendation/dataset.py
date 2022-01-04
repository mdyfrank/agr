'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)

Modified  on Nov 10, 2017, by Lianhai Miao
'''

import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class GDataset(object):

    def __init__(self, user_path, group_path, num_negatives, usr_num_negatives, g_m_d, g_m_d_neg, g_m_d_test,
                 g_m_d_neg_test):
        '''
        Constructor
        '''
        self.num_negatives = num_negatives
        self.usr_num_negatives = usr_num_negatives
        self.g_m_d = g_m_d
        self.g_m_d_neg = g_m_d_neg
        self.g_m_d_test = g_m_d_test
        self.g_m_d_neg_test = g_m_d_neg_test
        # user data
        self.user_trainMatrix = self.load_rating_file_as_matrix(user_path + "Train.txt")
        self.user_testRatings = self.load_rating_file_as_list(user_path + "Test.txt")
        self.user_testNegatives = self.load_negative_file(user_path + "Negative.txt")
        self.num_users, self.num_items = self.user_trainMatrix.shape
        # group data
        self.group_trainMatrix = self.load_rating_file_as_matrix(group_path + "Train.txt")
        self.group_testRatings = self.load_rating_file_as_list(group_path + "Test.txt")
        self.group_testNegatives = self.load_negative_file(group_path + "Negative.txt")
        # group-user data
        self.gu_testRatings = self.load_rating_file_as_list_gu(group_path + "Test.txt")
        self.gu_testNegatives = self.load_negative_file_gu(group_path + "Test.txt")



    def load_rating_file_as_list_gu(self, filename):
        '''
        ratingList: [[group, item, pos_user],...]
        '''
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                group, item = int(arr[0]), int(arr[1])
                if group in self.g_m_d_test:
                    pos_user_list = self.g_m_d_test[group]
                    for user in pos_user_list:
                        ratingList.append([group, item, user])
                line = f.readline()
        return ratingList

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file_gu(self, filename):
        '''
        negativeList: [[neg_user_1,neg_user_2,...],...]
        '''
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                negatives = []
                group, item = int(arr[0]), int(arr[1])
                if group in self.g_m_d_neg_test:
                    neg_user_list = self.g_m_d_neg_test[group]
                    for user in neg_user_list:
                        negatives.append(user)
                    negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_negative_file(self, filename):
        '''
        negativeList: [[neg_item_1,neg_item_2,...],...]
        '''
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    # def load_rating_file_as_matrix_group(self, filename):
    #     # Get number of users and items
    #     num_users, num_items = 0, 0
    #     with open(filename, "r") as f:
    #         line = f.readline()
    #         while line != None and line != "":
    #             arr = line.split(" ")
    #             u, i = int(arr[0]), int(arr[1])
    #             num_users = max(num_users, u)
    #             num_items = max(num_items, i)
    #             line = f.readline()
    #     # Construct matrix
    #     mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    #     with open(filename, "r") as f:
    #         line = f.readline()
    #         while line != None and line != "":
    #             arr = line.split(" ")
    #             if len(arr) > 2:
    #                 user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
    #                 if (rating > 0):
    #                     mat[user, item] = 1.0
    #             else:
    #                 user, item = int(arr[0]), int(arr[1])
    #                 mat[user, item] = 1.0
    #             line = f.readline()
    #     return mat

    def load_rating_file_as_matrix(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, pi_ni

    # def get_train_instances_group(self, train):
    #     user_input, pos_item_input, neg_item_input = [], [], []
    #     num_users = train.shape[0]
    #     num_items = train.shape[1]
    #     for (u, i) in train.keys():
    #         if u in self.g_m_d:
    #             # positive instance
    #             for _ in range(self.num_negatives):
    #                 pos_item_input.append(i)
    #             # negative instances
    #             for _ in range(self.num_negatives):
    #                 j = np.random.randint(num_items)
    #                 while (u, j) in train:
    #                     j = np.random.randint(num_items)
    #                 user_input.append(u)
    #                 neg_item_input.append(j)
    #     pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
    #     return user_input, pi_ni

    # a funtion return group_input, pu_nu
    def get_train_instances_gu(self, train):
        group_input, pos_user_input, neg_user_input, item_input = [], [], [], []
        for (g, i) in train.keys():
            pos_user_list = self.g_m_d[g]
            for u in pos_user_list:
                neg_user_input += self.g_m_d_neg[g]
                for _ in range(self.usr_num_negatives):
                    item_input.append(i)
                    pos_user_input.append(u)
                    group_input.append(g)
        pu_nu = [[pu, nu] for pu, nu in zip(pos_user_input, neg_user_input)]
        return group_input, item_input, pu_nu

    def get_user_dataloader(self, batch_size):
        user, positem_negitem_at_u = self.get_train_instances(self.user_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(positem_negitem_at_u))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return user_train_loader

    def get_group_dataloader(self, batch_size):
        group, positem_negitem_at_g = self.get_train_instances(self.group_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_negitem_at_g))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return group_train_loader

    def get_gu_dataloader(self, batch_size):
        group, positem_at_g, posuser_neguser_at_g = self.get_train_instances_gu(self.group_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_at_g),
                                   torch.LongTensor(posuser_neguser_at_g))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return group_train_loader
