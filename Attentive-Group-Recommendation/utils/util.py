'''
Created on Nov 10, 2017
Deal something

@author: Lianhai Miao
'''
import torch
from torch.autograd import Variable
import numpy as np
import math
import heapq


class Helper(object):
    """
        utils class: it can provide any function that we need
    """

    def __init__(self):
        self.timber = True

    def gen_group_member_dict(self, path, neg_num):
        g_m_d = {}
        g_m_d_neg = {}
        user_idx = set()
        with open(path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                a = line.split(' ')
                g = int(a[0])
                g_m_d[g] = []
                for m in a[1].split(','):
                    g_m_d[g].append(int(m))
                    user_idx.add(int(m))
                line = f.readline().strip()
            user_range = np.array(list(user_idx))
            for g in g_m_d:
                pos_idx_list = np.array(g_m_d[g])
                neg_range = np.setdiff1d(user_range, pos_idx_list)
                g_m_d_neg[g] = np.random.choice(neg_range, neg_num, replace=False).tolist()
        return g_m_d, g_m_d_neg

    def evaluate_model(self, model, testRatings, testNegatives, K, type_m):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs, acc = [], [], []
        if type_m in ['user','group']:
            for idx in range(len(testRatings)):
                (hr, ndcg) = self.eval_one_rating(model, testRatings, testNegatives, K, type_m, idx)
                hits.append(hr)
                ndcgs.append(ndcg)
            return (hits, ndcgs)
        elif type_m in ['user_group']:
            for idx in range(len(testRatings)):
                acc = self.eval_one_rating_gu(model, testRatings, testNegatives, 1, type_m, idx)
            return acc

    def eval_one_rating_gu(self, model, testRatings, testNegatives, K, type_m, idx):
        '''
        rating: [[group_1, item_1, pos_user_1],...]
        items: [[neg_user_1,neg_user_2,...,neg_user_n],...]
        '''
        rating = testRatings[idx]
        users = testNegatives[idx]
        g = rating[0]
        i = rating[1]
        gtUser = rating[2]
        users.append(gtUser)
        # Get prediction scores
        map_user_score = {}
        groups = np.full(len(users), g)
        items = np.full(len(users), i)

        groups_var = torch.from_numpy(groups)
        groups_var = groups_var.long()
        items_var = torch.from_numpy(items)
        items_var = items_var.long()

        users_var = torch.LongTensor(users)

        predictions = model(groups_var, users_var, items_var)

        for i in range(len(users)):
            user = users[i]
            map_user_score[user] = predictions.data.numpy()[i]
        users.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(K, map_user_score, key=map_user_score.get)
        hr = self.getHitRatio(ranklist, gtUser)
        ndcg = self.getNDCG(ranklist, gtUser)
        return (hr, ndcg)

    def eval_one_rating(self, model, testRatings, testNegatives, K, type_m, idx):
        '''
        rating: [[user_1, item_1],...]
        items: [[neg_1,neg_2,...,neg_n],...]
        '''
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u)

        users_var = torch.from_numpy(users)
        users_var = users_var.long()
        items_var = torch.LongTensor(items)
        if type_m == 'group':
            predictions = model(users_var, None, items_var)
        elif type_m == 'user':
            predictions = model(None, users_var, items_var)
        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.data.numpy()[i]
        items.pop()

        # Evaluate top rank list
        ranklist = heapq.nlargest(K, map_item_score, key=map_item_score.get)
        hr = self.getHitRatio(ranklist, gtItem)
        ndcg = self.getNDCG(ranklist, gtItem)
        return (hr, ndcg)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i + 2)
        return 0
