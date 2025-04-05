import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import torch
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

import matplotlib
matplotlib.rc('font', family='Arial', size=14, weight='bold')
matplotlib.rc('axes', titlesize=14, labelsize=14, titleweight='bold', labelweight='bold')
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rc('legend', fontsize=14)
matplotlib.rc('figure', titlesize=14)

ref_point = torch.zeros(2)
pass
cluster_lim = 20
cluster_lim_2 = 1e8

def pareto_dominance_check(df):
    y_err = df[0].values
    y_eng = df[1].values
    population = np.column_stack((y_err, y_eng))
    
    def dominates(p, q):
        return np.all(p <= q) and np.any(p < q)
    
    def fast_non_dominated_sort(population):
        S = [[] for _ in range(len(population))]
        n = np.zeros(len(population)) 
        fronts = [[]]
        rank = np.zeros(len(population)) 
        for p in range(len(population)):
            for q in range(len(population)):
                if dominates(population[p], population[q]):
                    S[p].append(q)
                elif dominates(population[q], population[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                fronts[0].append(p)
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        return fronts[:-1], rank

    def crowding_distance(population, front):
        distance = np.zeros(len(front))
        if len(front) == 0:
            return distance
        if len(front) <= 2:
            # 仅演示，未做完整实现
            return distance
        # 省略具体实现
        return distance
    
    fronts, rank = fast_non_dominated_sort(population)
    pass
    return fronts, population

def main():
    from multiprocessing import Semaphore
    with Semaphore(1) as sem:
        participants = []
        plt.figure(figsize=(10, 6))
        for ALGO in ["coflex","qnpargeo","qehvi","qnehvi","random","pabo"]:
            participants.append(ALGO)
        for algo in participants:
            file_path = ""
            df = None
            plt_color = ""
            print("acqu_algo:{}".format(algo))
            if algo == "qnehvi":
                file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/qnehvi/DeFiNES_SSS_os_ALGO_qNEHVI_TIME__02_13_15_55_imagenet_/train_output.csv'
                df = pd.read_csv(file_path, header=None)
                plt_color = "green"
            elif algo == "qnpargeo":
                file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/qnpargeo/DeFiNES_SSS_os_ALGO_qNParEGO_TIME__02_09_20_01_imagenet_/train_output.csv'
                df = pd.read_csv(file_path, header=None)
                plt_color = "red"
            elif algo == "coflex":
                file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/coflex_gen_2/DeFiNES_SSS_os_ALGO_Coflex_TIME__02_07_13_31_imagenet_/train_output.csv'
                df = pd.read_csv(file_path, header=None)
                plt_color = "blue"
            elif algo == "qehvi":
                file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/qehvi/DeFiNES_SSS_os_ALGO_qEHVI_TIME__02_15_09_54_imagenet_/train_output.csv'
                df = pd.read_csv(file_path, header=None)
                plt_color = "orange"
            elif algo == "nsga":
                file_path = ''
                df = pd.read_csv(file_path, header=None)
                plt_color = "black"
            elif algo == "pabo":
                file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/pabo/DeFiNES_SSS_os_ALGO_pabo_TIME__03_03_10_25_imagenet_/train_output.csv'
                df = pd.read_csv(file_path, header=None)
                plt_color = "grey"
            elif algo == "random":
                file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/random/DeFiNES_SSS_os_ALGO_random_TIME__02_16_01_17_imagenet_/train_output.csv'
                df = pd.read_csv(file_path, header=None)
                plt_color = "grey"

            assert df is not None
            y_err = df[0].values
            y_eng = df[1].values

            fronts, population = pareto_dominance_check(df)
            fronts_sel_index = []
            current_length = len(fronts_sel_index)
            fronts_sel_index.extend(fronts[0])
            front_points_less = population[fronts_sel_index]
            current_length = len(fronts_sel_index)
            front_index = 1
            while current_length < cluster_lim and front_index < len(fronts):
                for idx in fronts[front_index]:
                    if current_length < cluster_lim:
                        fronts_sel_index.append(idx)
                        current_length += 1
                    else:
                        break
                front_index += 1
            print("### Newly added ... ", fronts_sel_index)
            front_points = population[fronts_sel_index]
            ref_point[0] = 0
            ref_point[1] = 0
            while current_length < cluster_lim_2 and front_index < len(fronts):
                for idx in fronts[front_index]:
                    if current_length < cluster_lim_2:
                        fronts_sel_index.append(idx)
                        current_length += 1
                    else:
                        break
                front_index += 1
            front_points_all = population[fronts_sel_index]

            filtered_points = front_points[(front_points[:, 0] < 65) & (front_points[:, 1] < 1e4)]
            bd = DominatedPartitioning(ref_point=ref_point, Y=torch.tensor(filtered_points))
            volume = bd.compute_hypervolume().item()
            print(f'Pareto optimal region {np.log(volume):.2e}')

            filtered_points_all = front_points_all[(front_points_all[:, 0] < 65) & (front_points_all[:, 1] < 1e4)]
            if len(filtered_points) > 0:
                plt.scatter(
                    filtered_points[:, 0], 
                    filtered_points[:, 1], 
                    color=plt_color, 
                    s=50, 
                    label=algo, 
                    edgecolor='k', 
                    alpha=0.8
                )
            df_filtered = pd.DataFrame(filtered_points, columns=["Error", "EDP"])
            df_filtered.to_csv("filtered_points_" + algo + ".csv", index=False)

            print(f'### num points of {algo}', len(front_points_less[:, 0]))
            print()

        plt.title('Pareto comparison')
        plt.xlabel('Err(%)')
        plt.ylabel('EDP(µJ*s)')
        plt.legend()
        plt.grid(True)
        plt.savefig("pareto_comparison.png", dpi=300)
        plt.show()

if __name__ == '__main__':
    main()
