import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import torch
def process_data(df):
    cycle_time = []
    cycle_list = df[1].values
    cycle_his = []
    acc_time = []
    acc_list = df[0].values
    acc_his = []
    time_list = df[3].values
    cycle_low = np.sort(cycle_list)[-1]
    cycle_low_idx = np.argmax(cycle_list)
    for i in range(0, len(cycle_list)):
        if cycle_list[i] < cycle_low:
            cycle_low = cycle_list[i]
            cycle_low_idx = i
        cycle_time.append(time_list[i])
        cycle_his.append(cycle_low)
        # cycle_his.append(cycle_list[i])
    acc_low = np.sort(acc_list)[-1]
    acc_low_idx = np.argmax(acc_list)
    for i in range(0, len(acc_list)):
        if acc_list[i] < acc_low:
            acc_low = acc_list[i]
            acc_low_idx = i
        acc_time.append(time_list[i])
        acc_his.append(acc_low)
        # acc_his.append(acc_list[i])
    return cycle_his, cycle_low_idx, acc_his, acc_low_idx, cycle_time, acc_time
def main():
    from multiprocessing import Semaphore
    with Semaphore(1) as sem:
        participants = list()
        # plt.figure(figsize=(10, 10))
        # plt.style.use('ggplot')
        cycle_time_list = []
        cycle_his_list = []
        cycle_label_list = []
        cycle_color_list = []
        early_stop_cycle_idx_list = []
        # ****************************************** #
        acc_time_list = []
        acc_his_list = []
        acc_label_list = []
        acc_color_list = []
        early_stop_acc_idx_list = []
        # ****************************************** #
        edp_threshold = 1
        err_threshold = 55
        # ****************************************** #
        for ALGO in ["coflex", "qnpargeo", "qehvi", "qnehvi", "random","pabo"]: # "coflex", "qnpargeo", "qehvi", "qnehvi", "random", "pabo"]
            participants.append(ALGO)
        for algo in participants:
            file_path = ""
            df = None
            plt_color = ""
            print("acqu_algo:{}".format(algo))
            if algo == "qnehvi":
                file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/qnehvi/DeFiNES_SSS_os_ALGO_qNEHVI_TIME__02_13_15_55_imagenet_/opt_efficiency_analys.csv'
                df = pd.read_csv(file_path, header=None)
                plt_color = "green"
            elif algo == "qnpargeo":
                file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/qnpargeo/DeFiNES_SSS_os_ALGO_qNParEGO_TIME__02_09_20_01_imagenet_/opt_efficiency_analys.csv'
                df = pd.read_csv(file_path, header=None)
                plt_color = "red"
            elif algo == "coflex":
                file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/coflex_gen_2/DeFiNES_SSS_os_ALGO_Coflex_TIME__02_07_13_31_imagenet_/opt_efficiency_analys.csv'
                df = pd.read_csv(file_path, header=None)
                plt_color = "blue"
            elif algo == "qehvi":
                file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/qehvi/DeFiNES_SSS_os_ALGO_qEHVI_TIME__02_15_09_54_imagenet_/opt_efficiency_analys.csv'
                df = pd.read_csv(file_path, header=None)
                plt_color = "orange"
            elif algo == "nsga":
                pass
            elif algo == "pabo":
                file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/pabo/DeFiNES_SSS_os_ALGO_pabo_TIME__03_03_10_25_imagenet_/opt_efficiency_analys.csv'
                df = pd.read_csv(file_path, header=None, nrows=60)
                plt_color = "black"
            elif algo == "random":
                file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/random/DeFiNES_SSS_os_ALGO_random_TIME__02_16_01_17_imagenet_/opt_efficiency_analys.csv'
                df = pd.read_csv(file_path, header=None)
                plt_color = "grey"
            assert df is not None
            cycle_his, cycle_low_idx, acc_his, acc_low_idx, cycle_time, acc_time = process_data(df)
            # Set the thresholds for EDP and Err
            # Find the index where early stopping conditions are met
            early_stop_cycle_idx = next((i for i, val in enumerate(cycle_his) if val <= edp_threshold), None)
            early_stop_acc_idx = next((i for i, val in enumerate(acc_his) if val <= err_threshold), None)
            # Create a figure with two subplots
            cycle_time_list.append(cycle_time)
            cycle_his_list.append(cycle_his)
            cycle_label_list.append(algo)
            cycle_color_list.append(plt_color)
            early_stop_cycle_idx_list.append(early_stop_cycle_idx)
            acc_time_list.append(acc_time)
            acc_his_list.append(acc_his)
            acc_label_list.append(algo)
            acc_color_list.append(plt_color)
            early_stop_acc_idx_list.append(early_stop_acc_idx)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))
        # Plot Log(EDP) change
        import matplotlib.lines as mlines
        handles_ax1 = []
        labels_ax1 = []
        handles_ax2 = []
        labels_ax2 = []
        for i in range(len(cycle_time_list)):
            # -------------------- Ax1: Log(EDP) --------------------
            line1, = ax1.plot(cycle_time_list[i], np.log(cycle_his_list[i]), 
                            label=cycle_label_list[i], color=cycle_color_list[i], 
                            marker=".", alpha=0.7, markersize=2)
            handles_ax1.append(line1)
            labels_ax1.append(cycle_label_list[i])
            # ax1.annotate(f'{cycle_his_list[i][-1]:.2e}', 
            #             (cycle_time_list[i][-1] + 1, np.log(cycle_his_list[i][-1])), 
            #             textcoords="offset points", xytext=(5, 5), ha='center')
            if early_stop_cycle_idx_list[i] is not None:
                ax1.plot(cycle_time_list[i][early_stop_cycle_idx_list[i]], 
                        np.log(cycle_his_list[i][early_stop_cycle_idx_list[i]]), 
                        marker='*', markersize=10, color='gold', markeredgecolor='black')
                print("Cutoff iteration: {}".format(cycle_time_list[i][early_stop_cycle_idx_list[i]]))
            else:
                ax1.plot(cycle_time_list[i][-1], 
                        np.log(cycle_his_list[i][-1]), 
                        marker='s', markersize=5, color='red', markeredgecolor='black')
                print("Cutoff iteration: {}".format(cycle_time_list[i][-1]))
            final_res = mlines.Line2D([], [], color=cycle_color_list[i], marker='v', markersize=7,
                                    markeredgecolor='black', linestyle='None',
                                    label='Final result - {} - {:.2e}'.format(cycle_label_list[i], cycle_his_list[i][-1]))
            handles_ax1.append(final_res)
            labels_ax1.append(final_res.get_label())
            ax1.set_title('Log(EDP) change', fontsize=9)
            ax1.set_xlabel('Iteration', fontsize=9)
            ax1.set_ylabel('Log(EDP - µJ*s)', fontsize=9)
            ax1.grid(True)         
            # -------------------- Ax2: Err --------------------
            line2, = ax2.plot(acc_time_list[i], acc_his_list[i], 
                            label=acc_label_list[i], color=acc_color_list[i], 
                            marker="*", alpha=0.7, markersize=2)
            handles_ax2.append(line2)
            labels_ax2.append(acc_label_list[i])
            # ax2.annotate(f'{acc_his_list[i][-1]:.2e}', 
            #             (acc_time_list[i][-1] + 1, acc_his_list[i][-1]), 
            #             textcoords="offset points", xytext=(5, 5), ha='center')
            if early_stop_acc_idx_list[i] is not None:
                ax2.plot(acc_time_list[i][early_stop_acc_idx_list[i]], acc_his_list[i][early_stop_acc_idx_list[i]], 
                        marker='*', markersize=10, color='gold', markeredgecolor='black')
            else:
                ax2.plot(acc_time_list[i][-1], acc_his_list[i][-1], 
                        marker='s', markersize=5, color='red', markeredgecolor='black')
            final_res = mlines.Line2D([], [], color=acc_color_list[i], marker='v', markersize=7,
                                    markeredgecolor='black', linestyle='None',
                                    label='Final result - {} - {:.2e}'.format(acc_label_list[i], acc_his_list[i][-1]))
            handles_ax2.append(final_res)
            labels_ax2.append(final_res.get_label())
            ax2.set_title('Err change', fontsize=9)
            ax2.set_xlabel('Iteration', fontsize=9)
            ax2.set_ylabel('Err - %', fontsize=9)
            ax2.grid(True)
        gold_star = mlines.Line2D([], [], color='gold', marker='*', markersize=10,
                                    markeredgecolor='black', linestyle='None',
                                    label='Threshold reached')
        red_square = mlines.Line2D([], [], color='red', marker='s', markersize=5,
                                markeredgecolor='black', linestyle='None',
                                label='Threshold not reached')
        handles_ax1.extend([gold_star, red_square])
        labels_ax1.extend([gold_star.get_label(), red_square.get_label()])
        handles_ax2.extend([gold_star, red_square])
        labels_ax2.extend([gold_star.get_label(), red_square.get_label()])
        ax1.legend(handles_ax1, labels_ax1, fontsize=9, loc='upper right')
        ax2.legend(handles_ax2, labels_ax2, fontsize=9, loc='upper right')
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.show()

        err_records = []
        for i in range(len(acc_time_list)):
            for j in range(len(acc_time_list[i])):
                err_records.append({
                    'Algorithm': acc_label_list[i],
                    'Iteration': acc_time_list[i][j],
                    'Error': acc_his_list[i][j]
                })
        df_err = pd.DataFrame(err_records)

        edp_records = []
        for i in range(len(cycle_time_list)):
            for j in range(len(cycle_time_list[i])):
                edp_records.append({
                    'Algorithm': cycle_label_list[i],
                    'Iteration': cycle_time_list[i][j],
                    'LogEDP': np.log(cycle_his_list[i][j])
                })
        df_edp = pd.DataFrame(edp_records)

        df_err_grouped = df_err.groupby(['Algorithm', 'Iteration'], as_index=False).last()
        df_edp_grouped = df_edp.groupby(['Algorithm', 'Iteration'], as_index=False).last()

        df_err_grouped.to_csv('df_err_grouped.csv', index=False)
        df_edp_grouped.to_csv('df_edp_grouped.csv', index=False)

        err_threshold_iters = {}
        edp_threshold_iters = {}

        for i, algo in enumerate(acc_label_list):

            if early_stop_acc_idx_list[i] is not None:
                err_iter = acc_time_list[i][early_stop_acc_idx_list[i]]
            else:
                err_iter = 30  
            err_threshold_iters[algo] = err_iter

            if early_stop_cycle_idx_list[i] is not None:
                edp_iter = cycle_time_list[i][early_stop_cycle_idx_list[i]]
            else:
                edp_iter = 30  
            edp_threshold_iters[algo] = edp_iter

        print("Threshold iteration for Error (≤55%):")
        for algo, iters in err_threshold_iters.items():
            print(f"{algo}: {iters}")

        print("Threshold iteration for EDP (≤1 unit):")
        for algo, iters in edp_threshold_iters.items():
            print(f"{algo}: {iters}")

if __name__ == '__main__':
    main()