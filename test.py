#-*- coding:utf-8 _*-
import os, torch
import pdb
import argparse
from mmcv import Config
import numpy as np
from dataset import CIFAR10,MNIST
from models import MyModel
from trainers import Train_Model, Test_Model
from log import Logger
from EA import SFLA
from utils import dict_processing, save_params_dict
'''
    主函数
    总体实现所有的功能
    1. args, cfg, log
    2. 开始进化

'''

def parser():
    parse = argparse.ArgumentParser(description='Pytorch Cifar10 Training')
    # parse.add_argument('--local_rank',default=0,type=int,help='node rank for distributedDataParallel')
    parse.add_argument('--config','-c',default='./config/config.py',help='config file path')
    parse.add_argument('--ModelParams_H', '-mh', type=int, help='The highest model params')
    parse.add_argument('--ModelParams_L', '-ml', type=int, help='The lowest model params')
    parse.add_argument('--dataset', '-d', type=str, default='cifar10', help='Select the train dataset, \'mnist\' or \'cifar10\'')
    parse.add_argument('--evaluation', '-e', type=str, default='sur', help='Select the way of elaluation, \'sur\' or \'real\'')

    # parse.add_argument('--net','-n',type=str,required=True,help='input which model to use')
    # parse.add_argument('--net','-n',default='MyLenet5')
    # parse.add_argument('--pretrain','-p',action='store_true',help='Location pretrain data')
    # parse.add_argument('--resume','-r',action='store_true',help='resume from checkpoint')
    # parse.add_argument('--epoch','-e',default=None,help='resume from epoch')
    # parse.add_argument('--gpuid','-g',type=int,default=0,help='GPU ID')
    # parse.add_argument('--NumClasses','-nc',type=int,default=)
    args = parse.parse_args()
    return args

def My_SFLA():
    args = parser()
    cfg = Config.fromfile(args.config)

    # low = [0, 1, 10, 50]
    # high = [1,10, 50, 100]
    low = [50]
    high = [100]

    for i in range(len(low)):
        args.ModelParams_L, args.ModelParams_H = low[i], high[i]
        model_name = cfg.PARA.SFLA_params.model_name + '_' + args.evaluation \
                     + '_' + args.dataset + '_' + str(args.ModelParams_L) + '_' + str(args.ModelParams_H)
        print(model_name)
        log = Logger(cfg.PARA.utils_paths.log_path + model_name + '_log.txt', level='info')

        cifar10 = CIFAR10(batch_size=cfg.PARA.train_params.batch_size,
                          root=cfg.PARA.cifar10_params.root, download=False)
        train_loader, valid_loader = cifar10.Download_Train_Valid()
        sub_train_loder, sub_valid_loader = cifar10.Download_SubTrain_SubValid()
        test_loader = cifar10.Download_Test()

        sfla = SFLA(args=args, cfg=cfg, log=log,
                    dim= cfg.PARA.CNN_params.conv_num[1] + cfg.PARA.CNN_params.fc_num[1] + 2,
                    in_channels=cfg.PARA.cifar10_params.in_channels,
                    train_loader=sub_train_loder, valid_loader=sub_valid_loader,
                    model_name=model_name,
                    chromnum=cfg.PARA.SFLA_params.chromnum,
                    groupnum=cfg.PARA.SFLA_params.groupnum,
                    gens=cfg.PARA.SFLA_params.gens)

        # sfla.sur_update()

        '''Train the BestChrom again'''
        # BestChromi = sfla.BestChrom.to(torch.int)
        # BestChromi = torch.tensor([[3, 1, 2, 1, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0,]]) #0-1
        # BestChromi = torch.tensor([[4, 1, 1, 3, 2, 0, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]])  # 1-10
        # BestChromi = torch.tensor([[5, 3, 3, 3, 3, 1, 3, 2, 7, 7, 0, 0, 0, 0, 0, 0, 0, ]])  # 10-50
        BestChromi = torch.tensor([[4,  2,  2,  2,  3,  0,  5, 10, 10, 10, 10, 10,  0,  0,  0,  0,  0,]]) #50-100
        params_dict = sfla.chrom2paramsdict(BestChromi)
        save_params_dict(model_name, cfg.PARA.utils_paths.result_path, params_dict)


        print(params_dict)
        model = MyModel(params_dict=params_dict, chrom_i=-1, in_channels=sfla.in_channels)
        # pdb.set_trace()
        train_acc = Train_Model(model=model, train_loader=train_loader, valid_loader=valid_loader,
                                  args=args, cfg=cfg, log=log, model_name=model_name,
                                  epochs=cfg.PARA.train_params.best_epochs, is_datadict=True)

        test_acc = Test_Model(model=model, test_loader=test_loader,
                                args=args, cfg=cfg, log=log, model_name=model_name)

        dict_processing(model_name, cfg.PARA.utils_paths.result_path)


if __name__ == '__main__':
    My_SFLA()




