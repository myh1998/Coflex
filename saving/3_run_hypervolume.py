import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def main():

    participants = ["coflex","qnpargeo","qehvi","qnehvi","random","pabo"] # "coflex","qnpargeo","qehvi","qnehvi","random"

    plt.figure(figsize=(10, 6))

    for algo in participants:
        file_path = ""
        df = None
        plt_color = ""
        print("Processing algorithm: {}".format(algo))
        if algo == "qnehvi":
            file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/qnehvi/DeFiNES_SSS_os_ALGO_qNEHVI_TIME__02_13_15_55_imagenet_/hvs.csv'
            plt_color = "green"
            df = pd.read_csv(file_path, header=None)
        elif algo == "qnpargeo":
            file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/qnpargeo/DeFiNES_SSS_os_ALGO_qNParEGO_TIME__02_09_20_01_imagenet_/hvs.csv'
            plt_color = "red"
            df = pd.read_csv(file_path, header=None)
        elif algo == "coflex":
            file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/coflex_gen_2/DeFiNES_SSS_os_ALGO_Coflex_TIME__02_07_13_31_imagenet_/hvs.csv'
            plt_color = "blue"
            df = pd.read_csv(file_path, header=None)
        elif algo == "qehvi":
            file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/qehvi/DeFiNES_SSS_os_ALGO_qEHVI_TIME__02_15_09_54_imagenet_/hvs.csv'
            plt_color = "orange"
            df = pd.read_csv(file_path, header=None)
        elif algo == "random":
            file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/random/DeFiNES_SSS_os_ALGO_random_TIME__02_16_01_17_imagenet_/hvs.csv'
            plt_color = "grey"
            df = pd.read_csv(file_path, header=None)
        elif algo == "pabo":
            file_path = 'D:/OneDrive - Singapore University of Technology and Design/Saving/pabo/DeFiNES_SSS_os_ALGO_pabo_TIME__03_03_10_25_imagenet_/hvs.csv'
            plt_color = "purple"
            df = pd.read_csv(file_path, header=None, nrows=31)
        try:
            generations = np.arange(1, len(df) + 1)
            igd_values = df.iloc[:, 0].values  
            plt.plot(generations, igd_values, label=algo, color=plt_color)
        except Exception as e:
            print(f"Error processing {algo}: {e}")
            continue
    plt.xlabel('Generations', fontsize=12)
    plt.ylabel('Hypervolume', fontsize=12)
    plt.title('Hypervolume Iteration Process for Different Algorithms', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("igd_iterations_comparison.png", dpi=300)
    plt.show()
if __name__ == '__main__':
    main()
