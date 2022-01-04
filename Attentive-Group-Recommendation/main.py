'''
Created on Nov 10, 2017
Main function

@author: Lianhai Miao
'''
import sys

from model.agree import AGREE
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from time import time
from config import Config
from utils.util import Helper
from dataset import GDataset

from tqdm import tqdm

log_name = './log/' + str(time()) + '.txt'


# train the model
def training(model, train_loader, epoch_id, config, type_m):
    # user trainning
    learning_rates = config.lr
    # learning rate decay
    lr = learning_rates[0]
    if epoch_id >= 15 and epoch_id < 25:
        lr = learning_rates[1]
    elif epoch_id >= 20:
        lr = learning_rates[2]
    # lr decay
    if epoch_id % 5 == 0:
        lr /= 2

    # optimizer
    optimizer = optim.RMSprop(model.parameters(), lr)

    losses = []
    print('%s train_loader length: %d' % (type_m, len(train_loader)))
    if type_m in ['user', 'group']:
        for batch_id, (u, pi_ni) in tqdm(enumerate(train_loader)):
            # Data Load
            user_input = u
            pos_item_input = pi_ni[:, 0]
            neg_item_input = pi_ni[:, 1]
            # Forward
            if type_m == 'user':
                pos_prediction = model(None, user_input, pos_item_input)
                neg_prediction = model(None, user_input, neg_item_input)
            elif type_m == 'group':
                pos_prediction = model(user_input, None, pos_item_input)
                neg_prediction = model(user_input, None, neg_item_input)
            # Zero_grad
            model.zero_grad()
            # Loss
            loss = torch.mean((pos_prediction - neg_prediction - 1) ** 2)
            # record loss history
            losses.append(loss)
            # Backward
            loss.backward()
            optimizer.step()
    elif type_m in ['user_group']:
        for batch_id, (g, i, pu_nu) in tqdm(enumerate(train_loader)):
            group_input = g
            item_input = i
            pos_user_input = pu_nu[:, 0]
            neg_user_input = pu_nu[:, 1]
            pos_prediction = model(group_input, pos_user_input,item_input)
            neg_prediction = model(group_input, neg_user_input,item_input)
            model.zero_grad()
            # Loss
            loss = torch.mean((pos_prediction - neg_prediction - 1) ** 2)
            # record loss history
            losses.append(loss)
            # Backward
            loss.backward()
            optimizer.step()
    print('Iteration %d, loss is [%.4f ]' % (epoch_id, torch.mean(torch.stack(losses))))
    with open(log_name, 'a') as f:
        f.write('Iteration %d, loss is [%.4f ]\n' % (epoch_id, torch.mean(torch.stack(losses))))


def evaluation(model, helper, testRatings, testNegatives, K, type_m):
    model.eval()
    if type_m in ['user','group']:
        (hits, ndcgs) = helper.evaluate_model(model, testRatings, testNegatives, K, type_m)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        return hr, ndcg
    else:
        acc = helper.evaluate_model(model, testRatings, testNegatives, K, type_m)
        return acc
if __name__ == '__main__':
    # initial parameter class
    config = Config()

    # initial helper
    helper = Helper()

    # get the dict of users in group
    g_m_d, g_m_d_neg = helper.gen_group_member_dict(config.user_in_group_path, config.usr_num_negatives)
    g_m_d_test, g_m_d_neg_test = helper.gen_group_member_dict(config.user_in_group_path_test, config.user_num_negatives_test)
    # initial dataSet class
    dataset = GDataset(config.user_dataset, config.group_dataset, config.num_negatives,
                       config.usr_num_negatives, g_m_d, g_m_d_neg,g_m_d_test, g_m_d_neg_test)

    # get group number
    num_group = len(g_m_d)
    num_users, num_items = dataset.num_users, dataset.num_items

    # build AGREE model
    agree = AGREE(num_users, num_items, num_group, config.embedding_size, g_m_d, g_m_d_neg, config.drop_ratio)

    # config information
    print("AGREE at embedding size %d, run Iteration:%d, NDCG and HR at %d" % (
        config.embedding_size, config.epoch, config.topK))
    with open(log_name, 'a') as f:
        f.write("AGREE at embedding size %d, run Iteration:%d, NDCG and HR at %d\n" % (
            config.embedding_size, config.epoch, config.topK))
    # train the model
    for epoch in range(config.epoch):
        agree.train()
        # 开始训练时间
        t1 = time()
        # training(agree, dataset.get_user_dataloader(config.batch_size), epoch, config, 'user')

        training(agree, dataset.get_group_dataloader(config.batch_size), epoch, config, 'group')
        sys.exit()
        training(agree, dataset.get_gu_dataloader(config.batch_size), epoch, config, 'user_group')
        print("user and group training time is: [%.1f s]" % (time() - t1))
        with open(log_name, 'a') as f:
            f.write("user and group training time is: [%.1f s]\n" % (time() - t1))
        # evaluation - user-item
        t2 = time()
        u_hr, u_ndcg = evaluation(agree, helper, dataset.user_testRatings, dataset.user_testNegatives, config.topK,
                                  'user')
        print('User Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]' % (
            epoch, time() - t1, u_hr, u_ndcg, time() - t2))
        with open(log_name, 'a') as f:
            f.write('User Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]\n' % (
                epoch, time() - t1, u_hr, u_ndcg, time() - t2))
        # evaluation - group-item
        hr, ndcg = evaluation(agree, helper, dataset.group_testRatings, dataset.group_testNegatives, config.topK,
                              'group')
        print(
            'Group Iteration %d [%.1f s]: HR = %.4f, '
            'NDCG = %.4f, [%.1f s]' % (epoch, time() - t1, hr, ndcg, time() - t2))
        with open(log_name, 'a') as f:
            f.write(
                'Group Iteration %d [%.1f s]: HR = %.4f, '
                'NDCG = %.4f, [%.1f s]\n' % (epoch, time() - t1, hr, ndcg, time() - t2))
        # evaluation - group-user
        acc = evaluation(agree, helper, dataset.gu_testRatings, dataset.gu_testNegatives, config.topK,
                              'user_group')
        print(
            'Group Iteration %d [%.1f s]: ACC = %.4f, '
            '[%.1f s]' % (epoch, time() - t1, acc,time() - t2))
        with open(log_name, 'a') as f:
            f.write(
                'Group Iteration %d [%.1f s]: ACC = %.4f, '
                '[%.1f s]\n' % (epoch, time() - t1, acc, time() - t2))
    print("Done!")
