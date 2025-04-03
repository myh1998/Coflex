# Coflex
To run the code, please download [`COFleX_ho_5.zip`](https://sutdapac-my.sharepoint.com/:u:/g/personal/yinhui_ma_mymail_sutd_edu_sg/ERCSf7Dr0cxEkxaTk9g4mIgB2RjGA_DNLkkcsfRNVPTjWA?e=mZjcyM)

You can modify the project run parameters in `run_sss.py`. The default experiment run is set to NSGA with `popsize=100` and `gen=100`. For the NSGA `gen`, you can modify it in `termination=('n_gen', 100)`. Before running, you need to modify the `benchmark_root` to your absolute path in `COFleX_ho_5/Simulator/FRCN_Simulator.py` and `COFleX_ho_5/COFleX_Analysis/RBFleX/imageNet_SSS/Check_acc.py`.

The program supports using Bayesian optimization and NSGA for DeFiNES (which includes energy and latency analysis for various layer types) & Scalesim. Due to runtime constraints, it currently supports running on the Cifar100 dataset. To check the results, you can view the top 5 hardware and software configurations in the command line after the program has successfully completed. Additionally, you can find the corresponding results for DeFiNES and Scalesim in the `COFleX_result` directory.

When you want to run Extra hardware optimization or Extra software optimization, you can modify the `-opt_mode` parameter in `run_sss.py`. At this point, the `-fs` option will take effect, allowing you to provide the hardware and software configuration you wish to fix.
