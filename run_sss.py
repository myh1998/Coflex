import argparse
import pandas as pd
from Simulator.FRCN_Simulator import FRCN_Simulator
import sys
acc_code_path = "D:/COFleX/COFleX/COFleX_Analysis/RBFleX/imageNet_SSS"
sys.path.append(acc_code_path)
from COFleX_Analysis.RBFleX.imageNet_SSS.Check_acc import get_acc
import numpy as np
from tqdm import tqdm
import re
import os
import torch
import random
import time
def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)
def get_top1(df_output, df_input):
    if isinstance(df_output, torch.Tensor):
        df_output = df_output.cpu().numpy()
    if isinstance(df_input, torch.Tensor):
        df_input = df_input.cpu().numpy()
    df_output = pd.DataFrame(df_output)
    df_input = pd.DataFrame(df_input)
    err = list()
    if df_output.shape[1] < 2:
        raise ValueError("df_output must have at least two columns for 'err' and 'energy'.")
    err_df = df_output.iloc[:, 0]
    energy_df = df_output.iloc[:, 1]
    all_df = pd.concat([err_df, energy_df], axis=1)
    all_df.columns = ['err', 'energy']
    ref_point = np.array([0, 0])
    norm_all_data = all_df.apply(min_max_normalize)
    distance_list = list()
    # print("## This ", norm_all_data)
    for idx, data in norm_all_data.iterrows():
        point = np.array([data[0], data[1]])
        distance = np.linalg.norm(point - ref_point)
        distance_list.append(distance)
    distance_df = pd.DataFrame(distance_list)
    all_df = pd.concat([all_df, distance_df], axis=1)
    all_df.columns = ['err', 'energy', 'distance']
    idx_mindis = all_df['distance'].idxmin()
    min_row = all_df.loc[idx_mindis]
    top1_row = all_df.nsmallest(1, 'distance').values.tolist()
    # print("## This ")
    # print(min_row)
    return top1_row[0][0], top1_row[0][1]
def get_BestParam(df):
    all_df = df[['cycle', 'acc']].copy()
    ref_point = np.array([0, 1])
    norm_all_data = all_df.apply(min_max_normalize)
    distance_list = list()
    for idx, data in norm_all_data.iterrows():
        point = np.array([data[0], data[1]])
        distance = np.linalg.norm(point - ref_point)
        distance_list.append(distance)
    distance_df = pd.DataFrame(distance_list, columns=['distance'])
    all_df = pd.concat([all_df, distance_df], axis=1)
    all_df.columns = ['cycle', 'acc', 'distance']
    # print(all_df)
    # idx_mindis = all_df['distance'].idxmin()
    # min_row = df.loc[idx_mindis]
    # return min_row
def main(parser):
    from multiprocessing import Semaphore
    with Semaphore(1) as sem:
        global args
        args, unknown = parser.parse_known_args()
        optimized_components = {"X1": 0, "X2": 0, "X3": 0, "X4": 0, "X5": 0, "X6": 0}
        result_list = list()
        TOTAL_RUN_TIME = 0
        # Grid search Configs  
        gs_config = list()
        for N_HYPER in [10]: # 5,10,30
            for ACQU in ["Coflex","qNParEGO","qNEHVI","qEHVI","random"]: # "Coflex","qNParEGO","qNEHVI","qEHVI","random" 
                for ITERS in [30]: # 5, 15, 30, 45
                    for N_INIT in [100]: # 10,50,100,300
                        for BS in [10]: # 1,4,10
                            for H_ARCH in ["DeFiNES"]: # "ScaleSim", "DeFiNES"
                                gs_config.append([N_HYPER, ACQU, ITERS, N_INIT, BS, H_ARCH])
        for n_hyper, acqu_algo, iters, n_init_size, batch_size, hardware_arch in gs_config:
            print("n_hyper:{}, acqu_algo:{}, iters:{}, n_init_size:{}, batch_size:{}, hardware_arch:{}".format(n_hyper, acqu_algo, iters, n_init_size, batch_size, hardware_arch))
            frcn = FRCN_Simulator(IN_H=args.IN_H,
                                IN_W=args.IN_W,
                                opt_mode="co",
                                optimized_components=optimized_components,
                                set_hd_bounds=["1", "10", "64", "512"],
                                set_fs='0.0  0.0  0.0  0.0  0.0',
                                n_hyper=n_hyper, 
                                ref_score=-1000, 
                                acqu_algo=acqu_algo, 
                                iters=iters, 
                                n_init_size=n_init_size, 
                                batch_size=batch_size,
                                Hardware_Arch=hardware_arch, 
                                mapping="os",
                                TOTAL_RUN_TIME = TOTAL_RUN_TIME,
                                )  #ws, os, is
            for x in tqdm(range(1), ncols=80):
                print()
                print("### start: ", x)
                """
                Run Simulator
                """
                if acqu_algo == "qNEHVI":
                    acqu_algo_num = 0
                elif acqu_algo == "qEHVI":
                    acqu_algo_num = 1
                elif acqu_algo == "qNParEGO":
                    acqu_algo_num = 2
                elif acqu_algo == "Coflex":
                    acqu_algo_num = 3
                elif acqu_algo == "random":
                    acqu_algo_num = 4
                if acqu_algo_num in [0,1,2,3,4]:
                    train_obj, train_x = frcn.run()
                if hardware_arch == "DeFiNES":
                    top1_err, top1_energy = get_top1(train_obj, train_x)
                    result_list.append([n_hyper, acqu_algo_num, iters, n_init_size, batch_size, top1_err, top1_energy])
                    log_path = 'log' + '_' + str(acqu_algo) + '_' + str(x) + '.csv'
                    if not os.path.exists(log_path):
                        with open(log_path, 'w') as f:
                            np.savetxt(f, np.array(result_list), fmt='%s')
                    else:
                        with open(log_path, 'a') as f:
                            np.savetxt(f, np.array(result_list), fmt='%s')
            parameters_path = 'Parameter_search' + '_' + str(acqu_algo) + '_' + str(x) + '.csv'
            if not os.path.exists(parameters_path):
                df_results = pd.DataFrame(result_list, columns=['n_hyper', 'acqu_algo', 'iters', 'n_init_size', 'batch_size', 'top1_err', 'top1_energy'])
                df_results.to_csv(parameters_path, mode='w', header=True, index=False)
            else:
                df_results = pd.DataFrame(result_list, columns=['n_hyper', 'acqu_algo', 'iters', 'n_init_size', 'batch_size', 'top1_err', 'top1_energy'])
                df_results.to_csv(parameters_path, mode='a', header=False, index=False)
        # top config
        # get_BestParam(df_results)
        # print(get_BestParam(df_results))
if __name__ == '__main__':
    """
    initial set parameters
    """
    parser = argparse.ArgumentParser(description='DL2FRCN_Simulator')
    parser.add_argument('-ih','--IN-H', default=224, type=int, help='Height of input image for faster RCNN (default: 224)') # 224, 32
    parser.add_argument('-iw','--IN-W', default=224, type=int, help='Width of input image for faster RCNN (default: 224)') # 224, 32
    main(parser)