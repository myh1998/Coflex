import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def main():
    algorithms = ["coflex", "qnpargeo", "qehvi", "qnehvi", "random"]
    num_rows, num_cols = 3, 2  
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 9))
    # Flatten axes array in case it is multidimensional
    axes = axes.flatten()
    for idx, algo in enumerate(algorithms):
        file_path = ""
        df = None
        if algo == "qnehvi":
            file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/qnehvi/DeFiNES_SSS_os_ALGO_qNEHVI_TIME__02_13_15_55_imagenet_/opt_efficiency_analys.csv'
        elif algo == "qnpargeo":
            file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/qnpargeo/DeFiNES_SSS_os_ALGO_qNParEGO_TIME__02_09_20_01_imagenet_/opt_efficiency_analys.csv'
        elif algo == "coflex":
            file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/coflex_gen_2/DeFiNES_SSS_os_ALGO_Coflex_TIME__02_07_13_31_imagenet_/opt_efficiency_analys.csv'
        elif algo == "qehvi":
            file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/qehvi/DeFiNES_SSS_os_ALGO_qEHVI_TIME__02_15_09_54_imagenet_/opt_efficiency_analys.csv'
        elif algo == "random":
            file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/random/DeFiNES_SSS_os_ALGO_random_TIME__02_16_01_17_imagenet_/opt_efficiency_analys.csv'
        if file_path:
            df = pd.read_csv(file_path, header=None)
        if df is None:
            continue
        y_err = df[0].values
        y_eng = np.log(df[1].values)
        time_cost = df[2].values
        sc = axes[idx].scatter(y_err, y_eng, c=time_cost, cmap='viridis', label=algo, alpha=0.6)
        top50_indices = np.argsort(time_cost)[-50:]
        top50_err = y_err[top50_indices]
        top50_eng = y_eng[top50_indices]
        axes[idx].scatter(top50_err, top50_eng, marker='*', color='orange', label='Last 50 Time Cost', s=50)
        axes[idx].set_xlabel("Err - %", fontsize=8)
        axes[idx].set_ylabel("Log(EDP - µJ*s)", fontsize=8)
        axes[idx].set_title(f"Algorithm: {algo}", fontsize=8)
        axes[idx].legend()
        plt.colorbar(sc, ax=axes[idx], label="Time Cost")
    for i in range(len(algorithms), num_rows * num_cols):
        fig.delaxes(axes[i]) 
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    main()

