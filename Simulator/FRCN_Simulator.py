import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")
#warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
#warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
from Simulator.RBFleX import RBFleX
from Simulator.Computation import DL2
from Simulator.scale_sim.scale import scale as ScaleSim
from Simulator.defines.single_run import mainstage
from pymoo.algorithms.moo.nsga2 import NSGA2
import torch
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
# Multi Objective Bayesian Optimization
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize, standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.utils.sampling import sample_simplex
from botorch import fit_gpytorch_mll
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import (DominatedPartitioning,)
from collections import defaultdict
import time
import shutil
import sys
import pandas as pd
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from torch.quasirandom import SobolEngine
from pyswarm import pso
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from scipy.stats import norm
import random
import os
import pickle
import warnings
warnings.filterwarnings("ignore", message="A not p.d., added jitter")
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.double
else:
    DEVICE = "cpu"
    DTYPE = torch.double
tkwargs = {
    "dtype": DTYPE,
    "device": DEVICE,
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
clock_frequency_hz = 2.2 * 1e9
hvs = []
class FRCN_Simulator:
    def __init__(self, 
                IN_H, 
                IN_W,
                opt_mode,
                set_hd_bounds,
                set_fs,  
                optimized_components, 
                dataset="ImageNet", # cifar100, ImageNet, cifar10
                score_input_batch=8, 
                img_root=" ",
                benchmark="sss", # sss, 201
                benchmark_root="D:/COFleX/COFleX/NATS-sss-v1_0-50262-simple",
                # benchmark_root="D:/COFleX/COFleX/NATS-tss-v1_0-3ffb9-simple", # NATS-Bench/NATS-tss-v1_0-3ffb9-simple
                n_hyper=0, 
                ref_score=-0,
                iters=0, 
                mc_samples=128, 
                acqu_algo=" ", 
                batch_size=0, 
                num_restarts=10, 
                raw_samples=512, 
                n_init_size=0, 
                mapping=" ",
                hw='Meta_prototype_DF',
                Hardware_Arch=" ",
                TOTAL_RUN_TIME = 0,
        ):
        self.hw = hw
        self.set_opt_mode = opt_mode
        self.set_hd_bounds = set_hd_bounds
        self.set_fs = [int(float(x)) for x in set_fs.split()]
        self.set_acqu_algo = acqu_algo #
        self.set_iters = iters #
        self.set_n = n_init_size #
        self.set_Hardware_Arch = Hardware_Arch #
        self.set_mapping = mapping #
        self.current_iteration = 0
        self.dataset = dataset
        self.extra_rd_flag = False
        self.cluster_lim = 2
        # ------------------- Variables ----------------------- # 
        self.observation_theta = []
        self.observation_gamma = []
        self.beta = []
        self.IntPar = []
        self.DIntPar = []
        self.status = "NOM"
        self.err_list = []
        self.arch_list = []
        self.search_time_rec = None
        self.start_time = 0
        self.cost_time = 0
        self.estm_vist_time = 0
        self.GPU_H = 0
        self.TOTAL_RUN_TIME = TOTAL_RUN_TIME
        self.TOTAL_TRAIN_OBJ_REC = None
        self.iter = 0
        self.OPT_VS_TIME_REC = None
        # ------------------- Variables ----------------------- # 
        if self.extra_rd_flag == True:
            self.set_n = 1
        if self.dataset == "ImageNet":
            img_root="D:/OneDrive - Singapore University of Technology and Design/COFleX_imagenet/COFleX/dataset/ImageNet"
        if self.set_acqu_algo == 'NSGA':
            self.set_iters = 1
        print("___Configutations___", 
        self.set_acqu_algo, 
        self.set_opt_mode, 
        self.set_hd_bounds[0], 
        self.set_hd_bounds[1], 
        self.set_hd_bounds[2], 
        self.set_hd_bounds[3], 
        self.set_fs[0], 
        self.set_fs[1], 
        self.set_fs[2], 
        self.set_fs[3], 
        self.set_fs[4], 
        self.set_iters, 
        self.set_n, 
        self.set_Hardware_Arch, 
        self.set_mapping,
        self.extra_rd_flag,
        self.hw,
        self.dataset,
        img_root,
        )
        print('+++++++++++ Configuration +++++++++++')
        with torch.no_grad():
            #######################################
            # General config    
            #######################################
            print("[General config]")
            self.n_init_size = n_init_size
            self.image = torch.ones(1,3,IN_H, IN_W)
            print("\tInput resolution: [{}, {}]".format(IN_H, IN_W))
            print("\tAcquation function: {}".format(acqu_algo))
            print("\tBatch size for initial data generation: {}".format(n_init_size))
            print("\tDevice: {}".format(DEVICE))
            #######################################
            # Config for NAS     
            #######################################
            print("[RBFleX config]")
            self.batch_size_score = n_hyper # The number of batch images for hyperparameter detection algorithm
            self.benchmark = benchmark
            print("\tbatch images for HDA: {}".format(self.batch_size_score))
            #######################################
            # Config for DSE    
            #######################################
            print("[DSE config]")
            self.optimized_comp = optimized_components
            self.opt_architecture = 0
            self.hardware_components_values = {"X1":0, "X2":0, "X3":0, "X4":0, "X5":0, "X6":0}
            self.opt_params = [k for k, v in self.optimized_comp.items() if v == 0]
            self.not_opt_params = [k for k, v in self.optimized_comp.items() if v != 0]
            for key in self.not_opt_params:
                self.hardware_components_values[key] = (self.optimized_comp[key],self.optimized_comp[key])
            for key in self.opt_params:
                self.hardware_components_values[key] = (20,36)
            self.Hardware_Arch = Hardware_Arch
            if self.Hardware_Arch == "DL2":
                self.Num_HWopt = 6
            elif self.Hardware_Arch == "ScaleSim":
                self.Num_HWopt = 6
            elif self.Hardware_Arch == "DeFiNES":
                self.Num_HWopt = 6  # Number of HW varies to be optimized
                self.Num_SWopt = 5
            self.bounds_eng = [[0] * self.Num_HWopt, [0] * self.Num_HWopt]
            self.bounds_err = [[0] * self.Num_SWopt, [0] * self.Num_SWopt]
            print("Hardware Architecture: {}".format(self.Hardware_Arch))
            print('\tTo be optimized: {}'.format(self.opt_params))
            print('\tFixed HW params: {}'.format(self.not_opt_params))
            if self.Hardware_Arch == "DL2":
                if not mapping in ["rs", "ws"]:
                    print("[ERROR] mapping for DL2 supports only [rs, ws].")
            elif self.Hardware_Arch == "ScaleSim":
                if not mapping in ["os", "ws", "is"]:
                    print("[ERROR] mapping for systolic array supports only [os, ws, is].")
            elif self.Hardware_Arch == "DeFiNES":
                if not mapping in ["os", "ws", "is"]:
                    print("[ERROR] mapping for systolic array supports only [os, ws, is].")
            print('\tMapping: {} stationary'.format(mapping))
            #######################################
            #  Config for Multiple object baysian optimazation
            #######################################
            # NAS
            if self.benchmark == "sss":
                self.nas_dim = 5
                self.nas_obj = 1
                self.sf_lower_bound = 8
                self.sf_upper_bound = 64
                self.sf_bounds = torch.stack([torch.ones(self.nas_dim, **tkwargs), 8.0*torch.ones(self.nas_dim, **tkwargs)])
                self.sf_norm = torch.stack([self.sf_lower_bound*torch.ones(self.nas_dim, **tkwargs), self.sf_upper_bound*torch.ones(self.nas_dim, **tkwargs)])
                self.sf_standard_bounds = torch.stack([torch.zeros(self.nas_dim, **tkwargs), torch.ones(self.nas_dim, **tkwargs)])
            elif self.benchmark == "201":
                from design_space.models import get_search_spaces
                from design_space.policy import PolicyTopology
                self.space_201 = get_search_spaces('cell', 'nas-bench-201')
                self.policy = PolicyTopology(self.space_201 )
                self.nas_dim = 6
                self.nas_obj = 1
                self.sf_lower_bound = 0
                self.sf_upper_bound = 4
                self.sf_bounds = torch.stack([torch.zeros(self.nas_dim, **tkwargs), 4.0*torch.ones(self.nas_dim, **tkwargs)])
                self.sf_norm  = self.sf_bounds
                self.sf_standard_bounds = torch.stack([torch.zeros(self.nas_dim, **tkwargs), torch.ones(self.nas_dim, **tkwargs)])
            self.hd_dim = len(self.opt_params)
            if self.Hardware_Arch == "DL2":
                self.hd_obj = 2 #[energy and cycle]
                self.SCORE_IDX = 0
                self.ENERGY_IDX = 1
                self.CYCLE_IDX = 2
            elif self.Hardware_Arch == "ScaleSim":
                self.hd_obj = 1 #[cycle]
                self.SCORE_IDX = 0
                self.CYCLE_IDX = 1
            elif self.Hardware_Arch == "DeFiNES":
                self.hd_obj = 1  # [edp]
                self.ERROR_IDX = 0
                # self.ENERGY_IDX = 1
                # self.CYCLE_IDX = 2
                self.EDP_IDX = 1
            self.mobo_dim = self.nas_dim + self.hd_dim # how many obj to be optimized
            self.mobo_obj = self.nas_obj + self.hd_obj # how many output
            self.ref_point = torch.zeros(self.mobo_obj, **tkwargs)  # reference point
            self.ref_point[0] = ref_score
            if self.Hardware_Arch == "DL2":
                self.hd_lower_bound = [20,20]    #[0]: for PE array [1]: for memory
                self.hd_upper_bound = [60,60]    #[0]: for PE array [1]: for memory
            elif self.Hardware_Arch == "ScaleSim":
                self.hd_lower_bound = [1,10]     #[0]: for PE array [1]: for memory
                self.hd_upper_bound = [64,512]   #[0]: for PE array [1]: for memory
            elif self.Hardware_Arch == "DeFiNES":
                self.hd_lower_bound = [int(self.set_hd_bounds[0]), int(self.set_hd_bounds[2])]  # [0]: for PE array [1]: for memory
                self.hd_upper_bound = [int(self.set_hd_bounds[1]), int(self.set_hd_bounds[3])]  # [0]: for PE array [1]: for memory
            nk = 0
            self.hd_bounds = [[0]*self.Num_HWopt,[0]*self.Num_HWopt]
            for k, v in self.optimized_comp.items():
                if v == 0:
                    if k == "X1" or k == "X2":
                        self.hd_bounds[0][nk] = self.hd_lower_bound[0]
                        self.hd_bounds[1][nk] = self.hd_upper_bound[0]
                        self.bounds_eng[0][nk] = 1  # 1
                        self.bounds_eng[1][nk] = 64  # 64
                    else:
                        self.hd_bounds[0][nk] = self.hd_lower_bound[1]
                        self.hd_bounds[1][nk] = self.hd_upper_bound[1]
                        self.bounds_eng[0][nk] = 10
                        self.bounds_eng[1][nk] = 512
                else:
                    self.hd_bounds[0][nk] = v
                    self.hd_bounds[1][nk] = v
                    self.bounds_eng[0][nk] = v
                    self.bounds_eng[1][nk] = v
                nk += 1
            self.hd_bounds = torch.tensor(self.hd_bounds, **tkwargs)
            nk = 0
            self.hd_standard_bounds = torch.zeros(2, self.hd_dim, **tkwargs)
            self.hd_standard_bounds[1] = 1
            for k, v in self.optimized_comp.items():
                if not v == 0:
                    self.hd_standard_bounds[0][nk] = 1
                nk += 1
            nk = 0
            self.hd_norm = [[0]*self.Num_HWopt,[0]*self.Num_HWopt]
            for k, v in self.optimized_comp.items():
                if v == 0:
                    self.hd_norm[0][nk] = self.hd_lower_bound[1]
                    self.hd_norm[1][nk] = self.hd_upper_bound[1]
                    if k == "X1" or k == "X2":
                        self.hd_norm[0][nk] = self.hd_lower_bound[0]
                        self.hd_norm[1][nk] = self.hd_upper_bound[0]
                else:
                    self.hd_norm[0][nk] = 0
                    self.hd_norm[1][nk] = v
                nk += 1
            self.hd_norm = torch.tensor(self.hd_norm, **tkwargs)
            self.bounds = torch.cat((self.sf_bounds, self.hd_bounds),1)
            self.bounds_fornorm = torch.cat((self.sf_norm,self.hd_norm),1)
            self.bounds_forstard = torch.cat((self.sf_standard_bounds, self.hd_standard_bounds),1)
            self.BATCH_SIZE = batch_size
            self.NUM_RESTARTS = num_restarts if not SMOKE_TEST else 2
            self.RAW_SAMPLES = raw_samples if not SMOKE_TEST else 4
            self.N_BATCH = iters if not SMOKE_TEST else 10 # number of iteration
            self.MC_SAMPLES = mc_samples if not SMOKE_TEST else 16
            self.acqu_algo = acqu_algo
            self.bounds_eng = torch.tensor(self.bounds_eng, **tkwargs)
            self.bounds_err = torch.stack([8.0 * torch.ones(self.Num_SWopt, **tkwargs), 64.0 * torch.ones(self.Num_SWopt, **tkwargs)])
            self.bounds_fornorm = torch.cat((self.bounds_err, self.bounds_eng), 1)
            self.bounds_forstard_eng = torch.stack([torch.zeros(self.Num_HWopt, **tkwargs), torch.ones(self.Num_HWopt, **tkwargs)])
            self.bounds_forstard_err = torch.stack([torch.zeros(self.Num_SWopt, **tkwargs), torch.ones(self.Num_SWopt, **tkwargs)])
            self.bounds = torch.cat((self.bounds_err, self.bounds_eng), 1)
            self.bounds_forstard = torch.cat((self.bounds_forstard_err, self.bounds_forstard_eng), 1)
            #######################################
            # Initilize RBFleX, DSE, and Estimator    
            #######################################
            self.RBFleX = RBFleX(dataset, score_input_batch, img_root, benchmark, benchmark_root, self.batch_size_score)
            if self.Hardware_Arch == "DL2":
                self.AutoDSE = DL2(self.optimized_comp, mapping)
            elif self.Hardware_Arch == "ScaleSim":
                self.AutoDSE = ScaleSim(mapping)
            elif self.Hardware_Arch == "DeFiNES":
                self.AutoDSE = None
            self.mapping = mapping
    def delete_all_folders(self, path):
        if not os.path.exists(path):
            print(f"The path {path} does not exist.")
            return
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                try:
                    shutil.rmtree(item_path)
                except FileNotFoundError as e:
                    pass
            else:
                try:
                    os.remove(item_path)
                except FileNotFoundError as e:
                    pass
    def _rbflex_and_dse(self, image, candi_network, candi_hardparams):
        self.estm_vist_time += 1
        #######################
        #      RBFleX-NAS     
        #######################
        from COFleX_Analysis.RBFleX.imageNet_SSS.Check_acc import get_acc
        err = 100 - get_acc(candi_network)
        network_score, backbone, layers_dist, uid = self.RBFleX.run(image=image, arch=candi_network, hw=self.hw)
        # print("==> Current selected SW config: ", candi_network)
        #######################
        #         DSE         
        #######################
        if self.Hardware_Arch == "DL2":
            energy, cycle, latency = self.AutoDSE.run(image, layers_dist, candi_hardparams)
        elif self.Hardware_Arch == "ScaleSim":
            candi_hardparams[5] = 10 # bandwidth from ScaleSim
            print("==> Current selected Network arch: ", candi_network[0], candi_network[1], candi_network[2], candi_network[3], candi_network[4])
            print("==> Current selected HW arch: PE size_x {:.0f}, PE size_y {:.0f}, Mem for Ifmaps {:.0f}, "
                  "Mem for Ofmaps {:.0f}, Mem for W {:.0f}, Input Bandwidth {:.0f}".format(candi_hardparams[0],
                                                                                           candi_hardparams[1],
                                                                                           candi_hardparams[2],
                                                                                           candi_hardparams[3],
                                                                                           candi_hardparams[4],
                                                                                           candi_hardparams[5]))
            energy, cycle, latency = self.AutoDSE.run(layers_dist, candi_hardparams)
            # print("==> Current Energy & Latency: {:.6f} mJ, {:.6f} million cycles".format(energy, cycle))
        elif self.Hardware_Arch == 'DeFiNES':
            energy, cycle = 0, 0
            # set_uid(uid)
            # print("==> Current selected HW arch: PE size_x {:.0f}, PE size_y {:.0f}, Mem for Ifmaps {:.0f}, "
            #       "Mem for Ofmaps {:.0f}, Mem for W {:.0f}, Input Bandwidth {:.0f}".format(candi_hardparams[0],
            #                                                                                candi_hardparams[1],
            #                                                                                candi_hardparams[2],
            #                                                                                candi_hardparams[3],
            #                                                                                candi_hardparams[4],
            #                                                                                candi_hardparams[5]))
            answer = mainstage.run(uid, candi_hardparams, self.hw)
            for i in range(0, len(answer)):
                energy += answer[i][0].energy_total
                cycle += answer[i][0].latency_total1
            latency_in_seconds = cycle / clock_frequency_hz
            # print(
            #     "==> Current Energy & Cycle_Count & Latency & EDP: {:.6f} mJ, {:.6f} M_cycles ,{:.6f} seconds, {:.6f} µJ*s".format(
            #         energy * 1e-9, (cycle * 1e-6), latency_in_seconds,
            #         ((energy * 1e-9) * latency_in_seconds * 1e3)))  # change energy's unit from pJ to mJ 1e-12J -> 1e-3
        if self.Hardware_Arch == "DeFiNES":
            return err, (energy * 1e-9), (cycle * 1e-6), ((energy * 1e-9) * (cycle / clock_frequency_hz) * 1e3)
        elif self.Hardware_Arch == "ScaleSim":
            return err, energy, cycle 
    def _generate_initial_data(self, image, n):
        print("==> Generate initial data for optimization..")
        if self.acqu_algo == "Coflex":
            print("- Sampling method - LHS - ")
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.stats import qmc
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False 
            N = self.n_init_size  # 初始采样数量
            dim = 11  # 总维度

            # 拉丁超立方体采样
            sampler = qmc.LatinHypercube(d=dim)
            samples = sampler.random(n=N)

            sw_levels = np.arange(8, 65, 8)  
            sw_indices = np.floor(samples[:, 0:5] * len(sw_levels)).astype(int)
            sw_samples = sw_levels[sw_indices]

            hw_samples_part1 = np.floor(samples[:, 5:7] * 64).astype(int) + 1

            num_hw_options = 512 - 10 + 1
            hw_samples_part2 = np.floor(samples[:, 7:11] * num_hw_options).astype(int) + 10

            design_samples = np.hstack([sw_samples, hw_samples_part1, hw_samples_part2])
        else:
            print("- Sampling method - Sobol - ")

        def process_candidate(candidate, arch_func):
            """Helper function to process a candidate."""
            arch = arch_func(candidate)
            accelerator = candidate[5:]
            if self.Hardware_Arch == "DeFiNES":

                err, energy, cycle, EDP = self._rbflex_and_dse(image, arch, accelerator)
                
                return [err, EDP]
            
            elif self.Hardware_Arch == "ScaleSim":
                with torch.no_grad():
                    err, cycle = self._rbflex_and_dse(image, arch, accelerator)
                return [err, cycle]
        if self.benchmark == "sss":

            if self.acqu_algo == "Coflex":
                train_x = torch.tensor(design_samples, **tkwargs)
            else:
                train_x_hd = torch.floor(draw_sobol_samples(self.hd_bounds, n=n, q=1).squeeze(1))
                train_x_sf = 8 * torch.floor(draw_sobol_samples(self.sf_bounds, n=n, q=1).squeeze(1))
                train_x = torch.cat((train_x_sf, train_x_hd), 1)
            
            num_candidates = train_x.size(0)
            # Preallocate train_obj for performance
            train_obj = []
            for i in tqdm(range(num_candidates), ncols=80):
                candidate = train_x[i].int().tolist()
                arch_func = lambda c: '{}:{}:{}:{}:{}'.format(c[0], c[1], c[2], c[3], c[4])
                train_obj.append(process_candidate(candidate, arch_func))
            train_obj = torch.tensor(train_obj, **tkwargs)

        elif self.benchmark == "201":
            # Placeholder logic for benchmark 201
            # train_obj = torch.empty(0)
            # train_x = train_x_hd
            train_x_sf = 8 * torch.floor(draw_sobol_samples(self.sf_bounds, n=n, q=1).squeeze(1))
            train_x = torch.cat((train_x_sf, train_x_hd), 1)
            num_candidates = train_x.size(0)
            # Preallocate train_obj for performance
            train_obj = []
            for i in tqdm(range(num_candidates), ncols=80):
                candidate = train_x[i].int().tolist()
                arch_func = lambda c: '{}:{}:{}:{}:{}'.format(c[0], c[1], c[2], c[3], c[4])
                train_obj.append(process_candidate(candidate, arch_func))
            train_obj = torch.tensor(train_obj, **tkwargs)
        else:
            raise ValueError(f"Unsupported benchmark: {self.benchmark}")
        return train_x, train_obj
    def _initialize_model(self, train_x, train_obj, bounds):
        # define models for objective and constraint
        train_x = train_x.to(**tkwargs)
        train_xn = normalize(train_x, bounds)
        models = []
        train_obj = train_obj.to(**tkwargs)
        train_obj_stand = standardize(train_obj)
        for i in range(train_obj_stand.shape[-1]):
            train_y = train_obj_stand[..., i:i+1]
            models.append(SingleTaskGP(train_xn, train_y))
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model  
    def pareto_dominance_check(self, y_err, y_eng):
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
        fronts, _ = fast_non_dominated_sort(population)
        plt.figure(figsize=(10, 6))
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        # if len(fronts) <= 6:
        #     for i, front in enumerate(fronts):
        #         front_points = population[front]
        #         plt.scatter(front_points[:, 0], front_points[:, 1], color=colors[i % len(colors)], label=f'Front {i+1}', s=100, edgecolor='k', alpha=0.6)
        # else:
        #     group_size = 6
        #     groups = [fronts[i:i + group_size] for i in range(0, len(fronts), group_size)]
        #     selected_fronts = [group[0] for group in groups]
        #     for i, front in enumerate(selected_fronts):
        #         front_points = population[front]
        #         plt.scatter(front_points[:, 0], front_points[:, 1], color=colors[i % len(colors)], label=f'Front {i+1}', s=100, edgecolor='k', alpha=0.6)
        for i, front in enumerate(fronts):
            front_points = population[front]
            plt.scatter(front_points[:, 0], front_points[:, 1], color=colors[i % len(colors)], label=f'Front {i+1}', s=50, edgecolor='k', alpha=0.6)
        plt.title('Pareto Fronts Visualization', fontsize=9)
        plt.style.use('ggplot')
        plt.xlabel('Error (y_err)', fontsize=9)
        plt.ylabel('Energy (y_eng)', fontsize=9)
        plt.xticks(fontsize=9)
        plt.yticks(fontsize=9)
        plt.legend(loc='best', fontsize=9)
        plt.grid(True)
        plt.savefig('pareto fronts visualization.png', dpi=300)
        # plt.show()
        return fronts, population
    """[Todo]Check this program."""
    def _get_new_data(self, image, acqu_algo, model, train_x, train_obj, sampler):
        AF_flag = True
        if acqu_algo == "qNEHVI":
            candidates = self._optimize_qnehvi_and_get_observation(model, train_x, sampler)
        elif acqu_algo == "qEHVI":
            candidates = self._optimize_qehvi_and_get_observation(model, train_x, sampler)
        elif acqu_algo == "qNParEGO":
            candidates = self._optimize_qnparego_and_get_observation(model, train_x, sampler)
        elif acqu_algo == "Coflex":
            if self.status == "ERR":
                assert isinstance(train_x, np.ndarray), "train_x should be a NumPy array"
                assert isinstance(train_obj, np.ndarray), "train_obj should be a NumPy array"
                assert train_x.shape[1] == 5, "train_x should have 5 columns"
                assert train_obj.shape[1] == 1, "train_obj should have 1 column"
                assert train_x.shape[0] == train_obj.shape[0], "train_x and train_obj should have the same number of rows"
                assert train_x.ndim == 2, "train_x should be a 2D array"
                assert train_obj.ndim == 2, "train_obj should be a 2D array"
                # candidates = self._optimize_coflex_and_get_observation(model, train_x, sampler, self.bounds_err, self.bounds_forstard_err)
                kernel = C(1.0, (1e-8, 1e8)) * RBF(length_scale=32, length_scale_bounds=(8, 64))
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
                bounds_err = np.array([[8, 64]] * 5)
                candidates = []
                y_min_err = torch.min(torch.tensor(train_obj)).item()
                for i in range(self.BATCH_SIZE):
                    candidate = self.optimize_acquisition(gp, train_x, bounds_err, y_min_err).reshape(1, -1)
                    candidates.append(candidate)
                new_x = torch.tensor(np.vstack(candidates), **tkwargs)
            elif self.status == "ENG":
                candidates = self._optimize_coflex_and_get_observation(model, train_x, sampler, self.bounds_eng, self.bounds_forstard_eng)
            elif self.status == "NOM":
                candidates = self._optimize_qnparego_and_get_observation(model, train_x, sampler)
            pass
        elif acqu_algo == "NSGA":
            pass
        elif acqu_algo == "random":
            AF_flag = False
            new_x_hd = torch.floor(draw_sobol_samples(self.hd_bounds,n=self.BATCH_SIZE, q=1).squeeze(1))
            if self.benchmark == "sss":
                new_x_sf = 8*torch.floor(draw_sobol_samples(self.sf_bounds,n=self.BATCH_SIZE, q=1).squeeze(1))
                new_x = torch.cat((new_x_sf, new_x_hd), 1)
            elif self.benchmark == "201":
                new_x_sf = torch.floor(draw_sobol_samples(self.sf_bounds,n=self.BATCH_SIZE, q=1).squeeze(1))
                new_x = torch.cat((new_x_sf, new_x_hd), 1)
        else:
            print("Select correct acquation function from [qNEHVI, qEHVI, qNParEGO, random]")
            exit()
        if AF_flag:
            if self.status == "NOM":
                new_x =  torch.floor(unnormalize(candidates.detach(), bounds=self.bounds_fornorm))
            elif self.status == "ERR":
                # new_x =  torch.floor(unnormalize(candidates.detach(), bounds=self.bounds_err))
                pass
            elif self.status == "ENG":
                new_x =  torch.floor(unnormalize(candidates.detach(), bounds=self.bounds_eng))        
        if self.benchmark == "sss" or self.benchmark == "201":
            if self.status == "NOM":
                sf_new_x = new_x[:, :5]
                sf_new_x = torch.round(sf_new_x/8)*8
                new_x[:, :5] = sf_new_x
            elif self.status == "ERR":
                # new_x =  torch.floor(unnormalize(candidates.detach(), bounds=self.bounds_err))
                # new_x = torch.round(new_x/8)*8
                assert new_x.shape[1] == 5, "new_x should have 5 columns"
                assert new_x.shape[0] == self.BATCH_SIZE, "new_x should have the same number of rows as self.BATCH_SIZE"
                assert new_x.ndim == 2, "new_x should be a 2D tensor"
                # pass
            elif self.status == "ENG":
                new_x =  torch.floor(unnormalize(candidates.detach(), bounds=self.bounds_eng))
        if self.status == "NOM":
            with torch.no_grad():
                new_obj = []
                for candidate in new_x.tolist():
                    if self.Hardware_Arch == "DeFiNES":
                        if self.benchmark == "sss" or self.benchmark == "201":
                            candidate = list(map(int, candidate))
                            arch = '{}:{}:{}:{}:{}'.format(candidate[0], candidate[1], candidate[2], candidate[3],
                                                        candidate[4])
                            # print("this network arch: " + str(candidate[0]) + "," + str(candidate[1]) + "," + str(
                            #     candidate[2]) + "," + str(candidate[3]) + "," + str(candidate[4]))
                            accelerator = candidate[5:]
                        err, energy, cycle, EDP = self._rbflex_and_dse(image, arch, accelerator)
                        self.cost_time = time.time() - self.start_time
                        if self.search_time_rec is None:
                            self.search_time_rec = torch.tensor([[err, EDP, self.cost_time, self.iter]], **tkwargs)
                        else:
                            self.search_time_rec = torch.cat(
                                (self.search_time_rec, 
                                torch.tensor([[err, EDP, self.cost_time, self.iter]], **tkwargs)
                                )
                                , dim=0)
                    elif self.Hardware_Arch == "ScaleSim":
                        if self.benchmark == "sss":
                            candidate = list(map(int, candidate))
                            arch = '{}:{}:{}:{}:{}'.format(candidate[0], candidate[1], candidate[2], candidate[3],
                                                        candidate[4])
                            # print("this network arch: " + str(candidate[0]) + "," + str(candidate[1]) + "," + str(
                            #     candidate[2]) + "," + str(candidate[3]) + "," + str(candidate[4]))
                            accelerator = candidate[5:]
                        err, cycle = self._rbflex_and_dse(image, arch, accelerator)
                    if self.Hardware_Arch == "DL2":
                        new_obj.append([err, energy, cycle])
                    elif self.Hardware_Arch == "ScaleSim":
                        new_obj.append([err, cycle])
                    elif self.Hardware_Arch == "DeFiNES":
                        new_obj.append([err, EDP])
            new_obj = torch.tensor(new_obj, **tkwargs)
            # print("## The candidates ", new_x, new_obj)
            return new_x, new_obj
        elif self.status == "ERR":
            for candidate in new_x.tolist():
                arch = '{}:{}:{}:{}:{}'.format(int(candidate[0]), int(candidate[1]), int(candidate[2]), int(candidate[3]), int(candidate[4]))
                from COFleX_Analysis.RBFleX.imageNet_SSS.Check_acc import get_acc
                err = 100 - get_acc(arch)
                self.err_list.append(err)
                self.arch_list.append(arch)
            self.err_list = self.err_list[-(self.BATCH_SIZE):]
            self.arch_list = self.arch_list[-(self.BATCH_SIZE):]
            return new_x, torch.tensor(self.err_list).to(**tkwargs)
            pass
        elif self.status == "ENG":
            MIN_IDX = self.err_list.index(min(self.err_list))
            ADDIT_ERR = torch.tensor(([[[int(x) for x in self.arch_list[MIN_IDX].split(":")]]*10][0]), **tkwargs)
            ADDIT_ENG = new_x
            new_x = torch.cat((ADDIT_ERR, ADDIT_ENG), dim=1)
            with torch.no_grad():
                new_obj = []
                for candidate in new_x.tolist():
                    if self.Hardware_Arch == "DeFiNES":
                        if self.benchmark == "sss" or self.benchmark == "201":
                            candidate = list(map(int, candidate))
                            arch = '{}:{}:{}:{}:{}'.format(candidate[0], candidate[1], candidate[2], candidate[3],
                                                        candidate[4])
                            # print("this network arch: " + str(candidate[0]) + "," + str(candidate[1]) + "," + str(
                            #     candidate[2]) + "," + str(candidate[3]) + "," + str(candidate[4]))
                            accelerator = candidate[5:]
                        err, energy, cycle, EDP = self._rbflex_and_dse(image, arch, accelerator)
                    if self.Hardware_Arch == "DeFiNES":
                        new_obj.append([EDP])
            new_obj = torch.tensor(new_obj, **tkwargs).T.squeeze(0)
            return new_x[:, 5:], new_obj.to(**tkwargs)
    """[Todo] Solve the error. Float for the destination and Double for the source """
    def _optimize_qnehvi_and_get_observation(self, model, train_x, sampler):
        # partition non-dominated space into disjoint rectangles
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point,
            X_baseline=normalize(train_x, self.bounds_fornorm),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds_forstard,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        
        return candidates
    def _optimize_qehvi_and_get_observation(self, model, train_x, sampler):
        # partition non-dominated space into disjoint rectangles
        with torch.no_grad():
            pred = model.posterior(normalize(train_x, self.bounds_fornorm)).mean
        partitioning = FastNondominatedPartitioning(
            ref_point=self.ref_point, 
            Y=pred,
        )
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=self.ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds_forstard,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        return candidates
    def _optimize_qnparego_and_get_observation(self, model, train_x, sampler):
        train_x = normalize(train_x, self.bounds_fornorm)
        with torch.no_grad():
            pred = model.posterior(train_x).mean
        acq_func_list = []
        for _ in range(self.BATCH_SIZE):
            weights = sample_simplex(self.mobo_obj, **tkwargs).squeeze()
            objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
            acq_func = qNoisyExpectedImprovement(  # pyre-ignore: [28]
                model=model,
                objective=objective,
                X_baseline=train_x,
                sampler=sampler,
                prune_baseline=True,
            )
            acq_func_list.append(acq_func)
        # optimize
        candidates, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=self.bounds_forstard,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        return candidates
    def _optimize_coflex_and_get_observation(self, model, train_x, sampler, bounds, bounds_forstard):
        train_x = torch.tensor(train_x, **tkwargs)
        train_x = normalize(train_x, bounds)
        with torch.no_grad():
            pred = model.posterior(train_x).mean
        acq_func_list = []
        for _ in range(self.BATCH_SIZE):
            weights = sample_simplex(1, **tkwargs)[0] 
            objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
            acq_func = qNoisyExpectedImprovement(  # pyre-ignore: [28]
                model=model,
                objective=objective,
                X_baseline=train_x,
                sampler=sampler,
                prune_baseline=True,
            )
            acq_func_list.append(acq_func)
        # optimize
        candidates, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=bounds_forstard,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        return candidates
    def _optimize_nsga_and_get_observation(self, train_x, train_obj, bounds_fornorm, image, Name_flag):
        # from pymoo.config import Config
        # Config.warnings['not_compiled'] = False
        # from pymoo.optimize import minimize
        # from pymoo.problems import get_problem
        # from pymoo.util.ref_dirs import get_reference_directions
        # from pymoo.visualization.scatter import Scatter
        # import numpy as np
        # from pymoo.core.callback import Callback
        # from pymoo.indicators.hv import HV
        # from pymoo.core.population import Population
        # from pymoo.core.problem import Problem
        # from pymoo.core.individual import Individual
        # from pymoo.core.evaluator import Evaluator
        # ##################################
        # import random
        # from deap import base, creator, tools, algorithms
        # train_x = train_x.int()
        # class CustomProblem(Problem):
        #     def __init__(self, train_x, train_obj, bounds_fornorm, image, Name_flag, rbflex_method, extra_rd_flag):
        #         self.bounds_fornorm = bounds_fornorm.numpy()
        #         self.train_x = train_x.numpy()
        #         self.train_obj = train_obj.numpy()
        #         self.rbflex_method = rbflex_method
        #         self.image = image
        #         self.Name_flag = Name_flag
        #         self.extra_rd_flag = extra_rd_flag
        #         super().__init__(n_var=11,  # column of train_x as the numbers of vars
        #                         n_obj=2,  # two objective targets
        #                         n_constr=0,  # no constrain,
        #                         xl=np.array([1] * 5 + [1] * 2 + [10] * 4),
        #                         xu=np.array([8] * 5 + [64] * 2 + [512] * 4)
        #                         )
        #     def _evaluate(self, X, out, *args, **kwargs):
        #         def get_network_score_edp(x):
        #             x = list(map(int, x))
        #             arch = '{}:{}:{}:{}:{}'.format(x[0], x[1], x[2], x[3], x[4])
        #             accelerator = x[5:]
        #             err, cycle = self.rbflex_method(self.image, arch, accelerator, self.Name_flag, None)
        #             return err, cycle
        #         def get_acc(x):
        #             from COFleX_Analysis.RBFleX.imageNet_SSS.Check_acc import get_acc
        #             arch = '{}:{}:{}:{}:{}'.format(int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]))
        #             return get_acc(arch)
        #         X[:, :5] = (np.round(X[:, :5]) * 8).astype(int)
        #         X[:, 5:] = X[:, 5:].astype(int)
        #         f1 = []
        #         f2 = []
        #         # f3 = []
        #         for q in range(X.shape[0]):
        #             err, cycle = get_network_score_edp(X[q, :])
        #             # -------------------
        #             f1.append(cycle)
        #             f2.append(err)
        #             # -------------------
        #         out["F"] = np.column_stack([f1, f2])
        # problem = CustomProblem(train_x, train_obj, bounds_fornorm, image, Name_flag, self._rbflex_and_dse, self.extra_rd_flag)
        # if self.extra_rd_flag == True:
        #     pass
        # else:
        #     pop = Population.new("X", train_x)
        #     print("___Compare___", pop[0], train_obj[0])
        #     train_x[:, :5] = (torch.round(train_x[:, :5]) / 8).int()
        # Evaluator().eval(problem, pop)
        # # print("___Compare___", pop[0].F, train_obj[0])
        # algorithm = NSGA2(
        #     pop_size=len(train_x),
        #     crossover_prob=0.8,
        #     mutation_prob=0.2,
        #     eliminate_duplicates=True,
        #     sampling=pop
        #     )
        # res = minimize(problem,
        #             algorithm,
        #             termination=('n_gen', 5),
        #             seed=1,
        #             verbose=True,
        #             save_history=True,
        #             )
        # # callback.plot_hypervolume(res.F)
        # val = [e.opt.get("F")[0] for e in res.history]
        # generations = np.arange(len(val))
        # cycle = [v[0] for v in val]
        # acc = [v[1] for v in val]
        # # ---------------------- Hypervolume --------------------- #
        # y_tensor = torch.tensor(val)
        # bd = DominatedPartitioning(ref_point=self.ref_point, Y=y_tensor)
        # volume = bd.compute_hypervolume().item()
        # hvs.append(volume)
        # print("### The final hypervolume: ", hvs)
        # # ---------------------- Hypervolume --------------------- #
        # # Create two subplots
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        # # Plot cycle on the first subplot
        # ax1.set_title('Cycle over Generations')
        # ax1.set_xlabel('Generation')
        # ax1.set_ylabel('Cycle')
        # ax1.plot(generations, cycle, color='tab:orange')
        # ax1.tick_params(axis='y', labelcolor='tab:orange')
        # # Plot accuracy on the second subplot
        # ax2.set_title('Accuracy over Generations')
        # ax2.set_xlabel('Generation')
        # ax2.set_ylabel('Accuracy')
        # ax2.plot(generations, acc, color='tab:blue')
        # ax2.tick_params(axis='y', labelcolor='tab:blue')
        # # Adjust layout
        # fig.tight_layout(pad=3.0)
        # fig.suptitle('Pareto Front Convergence', y=1.02)
        # # Save the plot
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # target_dir = os.path.join(current_dir, '..', 'COFleX_result', f'{self.Hardware_Arch}_SSS_os', 'plot_saving')
        # os.makedirs(target_dir, exist_ok=True)
        # target_file_path = os.path.join(target_dir, f'pareto_front_related_f{self.current_iteration}.png')
        # plt.savefig(target_file_path, dpi=300, bbox_inches='tight')
        # fig, ax = plt.subplots(figsize=(5, 5))
        # ax.scatter(res.F[:, 0], res.F[:, 1])
        # ax.set_xlabel('f1')
        # ax.set_ylabel('f2')
        # ax.set_title('f1 vs f2')
        # ax.grid(True)
        # target_file_path = os.path.join(target_dir, f'pareto_front_2d_{self.current_iteration}.png')
        # plt.savefig(target_file_path, dpi=300)
        # # plt.show()
        # target_file_path = os.path.join(target_dir, f'output_res_f{str(time.time())}.csv')
        # df = pd.DataFrame(res.F)
        # df.to_csv(target_file_path, index=False)
        # candidates = torch.tensor(np.hstack([(np.round(res.X[:, :5]) * 8).astype(int), res.X[:, 5:].astype(int)]))
        # print("___This candidates___", self.current_iteration, candidates)
        pass
        # return candidates
    """[Todo] change negative to positive"""
    def plots(self, hsv_list, train_obj):
        fig = plt.figure()
        ax_re = fig.add_subplot(1, 2, 1)
        train_obj = train_obj.cpu().numpy()
        ax_re.scatter(
            train_obj[:, 0], -1 * train_obj[:, 1], alpha=0.8
        )
        ax_re.set_title("AF: {} H-Volume: {}".format(self.acqu_algo, hsv_list[-1]))
        ax_re.set_xlabel("Network Score")
        ax_re.set_ylabel("Cycle")

        ax_2d = fig.add_subplot(1, 2, 2)
        ax_2d.plot(hsv_list)
        ax_2d.set_xlabel("Iteration")
        ax_2d.set_ylabel("H-volume")

        plt.subplots_adjust(wspace=0.4)
        save_dir = "COFleX_result/" + self.Hardware_Arch + "/plot_saving/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + "hypervolume.png", dpi=300)
        # plt.show()
    def fast_non_dominated_sort(self, P):
        F = defaultdict(list)
        for p in P:
            p.S = []
            p.n = 0
            for q in P:
                if p < q:  # if p dominate q
                    p.S.append(q)  # Add q to the set of solutions dominated by p
                elif q < p:
                    p.n += 1  # Increment the domination counter of p
            if p.n == 0:
                p.rank = 1
                F[1].append(p)
        i = 1
        while F[i]:
            Q = []
            for p in F[i]:
                for q in p.S:
                    q.n = q.n - 1
                    if q.n == 0:
                        q.rank = i + 1
                        Q.append(q)
            i = i + 1
            F[i] = Q
        return F
    def crowding_distance_assignment(self, L):
        """传进来的参数应该是L = F(i)，类型是List，且 objective 是 ndarray"""
        l = len(L)  # number of solutions in F
        for i in range(l):
            L[i].distance = 0  # initialize distance
        num_objectives = L[0].objective.shape[0]  # number of objectives
        for m in range(num_objectives):
            # Sort using each objective value
            L.sort(key=lambda x: x.objective[m])
            # Boundary points
            L[0].distance = float('inf')
            L[l - 1].distance = float('inf')
            f_max = L[l - 1].objective[m]
            f_min = L[0].objective[m]
            # Avoid division by zero
            if f_max == f_min:
                # print(f"Objective {m}: Max value {f_max} equals Min value {f_min}, skipping.")
                continue
            for i in range(1, l - 1):  # for all other points
                L[i].distance += (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)
    def plot_P(self, P):
        cmap = plt.get_cmap("tab10")  
        colors = [cmap(i) for i in range(len(P))]  
        plt.figure(figsize=(10, 6))
        for i, t in enumerate(P):  
            X = [ind.objective[0] for ind in t]
            Y = [ind.objective[1] for ind in t]
            plt.scatter(X, Y, color=colors[i], alpha=0.5, label=f"Layer {i+1}")  
        plt.xlabel('F1')
        plt.ylabel('F2')
        plt.legend()
        plt.title("Population Distribution Across Layers")
        # plt.show()
        plt.savefig('pareto fronts visualization.png', dpi=300)
        plt.close()
    def run(self):
        print('+++++++++++ Optimization +++++++++++')
        print('==> Reproducibility..')

        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)

        # with open("random_state.pkl", "wb") as f:
        #     pickle.dump({
        #         "numpy_state": np.random.get_state(),
        #         "torch_state": torch.get_rng_state(),
        #         "python_random_state": random.getstate()
        #     }, f)

        load_path = 'D:/COFleX/COFleX/COFleX_saving/_INIT_DATA_SIZE100_TIME__01_21_07_16_'
        # load_path = 'D:/COFleX/COFleX/COFleX_saving/xxx_xxx'
        if os.path.exists(load_path):
            train_x_path = f"{load_path}/train_input.csv"
            train_obj_path = f"{load_path}/train_output.csv"
            if os.path.exists(train_x_path) and os.path.exists(train_obj_path):
                try:
                    train_x = pd.read_csv(train_x_path, nrows=100, header=None).iloc[:, :11]  # 取前 11 列
                    train_obj = pd.read_csv(train_obj_path, nrows=100, header=None).iloc[:, :2]  # 取前 2 列
                    train_x = train_x.values
                    train_obj = train_obj.values
                    train_x = torch.tensor(train_x, **tkwargs)
                    train_obj = torch.tensor(train_obj, **tkwargs)
                    print("Data loaded successfully:")
                    print(f"train_x shape: {train_x.shape}, train_obj shape: {train_obj.shape}")
                except Exception as e:
                    print(f"Error loading data: {e}")
            else:
                pass
        else:
            print("Required files not found at the specified paths.")
            print("-- No file found at the specified path, save another --")
            train_x, train_obj = self._generate_initial_data(self.image, n=self.n_init_size)
            save_path = 'COFleX_saving/' + '_INIT_DATA_SIZE' + str(self.n_init_size) + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_")
            os.makedirs(save_path, exist_ok=True)
            try:
                np.savetxt(os.path.join(save_path, 'train_input.csv'), train_x.cpu().numpy())
                np.savetxt(os.path.join(save_path, 'train_output.csv'), train_obj.cpu().numpy())
            except Exception as e:
                print(f"Error saving files: {e}")
        self.cost_time = time.time() - self.start_time
        self.OPT_VS_TIME_REC = torch.cat((train_obj, torch.full((train_obj.shape[0], 1), self.cost_time, **tkwargs)), dim=1)
        assert self.OPT_VS_TIME_REC.shape[1] == 3, "OPT_VS_TIME_REC should have 3 columns"
        self.TOTAL_TRAIN_OBJ_REC = train_obj
        self.start_time = time.time()
        train_x_err = train_x[:, :5].cpu().numpy() # θ
        train_x_eng = train_x[:, 5:].cpu().numpy() # Γ
        train_obj_err = train_obj[:, 0]
        train_obj_eng = train_obj[:, 1]
        train_err_recording = np.array([]) # Dperf
        train_eng_recording = np.array([]) # Deng
        IntPar_recording = []
        self.observation_theta.append(train_obj_err) # O(θ)
        self.observation_gamma.append(train_obj_eng) # O(Γ)
        train_err_recording = np.append(train_err_recording, train_x_err) # Dperf
        train_eng_recording = np.append(train_eng_recording, train_x_eng) # Deng
        train_x = torch.tensor(np.hstack((train_err_recording[:(len(train_err_recording) // 5) * 5].reshape(-1, 5), train_eng_recording[:(len(train_eng_recording) // 6) * 6].reshape(-1, 6))))
        train_obj = torch.cat((torch.cat(self.observation_theta, dim=0).unsqueeze(1), torch.cat(self.observation_gamma, dim=0).unsqueeze(1)), dim=1)
        mll, model = self._initialize_model(train_x, train_obj, self.bounds_fornorm)
        mll_err, model_err = self._initialize_model(train_x[:, 0:5], train_obj[:, 0:1], self.bounds_err)
        mll_eng, model_eng = self._initialize_model(train_x[:, 5:], train_obj[:, 1:], self.bounds_eng)
        hvs = []
        # Reference points
        min_values, _ = torch.min(train_obj, dim=0)
        if self.Hardware_Arch == "DL2":
            self.ref_point[self.SCORE_IDX] = min_values[self.SCORE_IDX]
            self.ref_point[self.ENERGY_IDX] = min_values[self.ENERGY_IDX]
            self.ref_point[self.CYCLE_IDX] = min_values[self.CYCLE_IDX]
        elif self.Hardware_Arch == "ScaleSim":
            self.ref_point[self.SCORE_IDX] = min_values[self.SCORE_IDX]
            self.ref_point[self.CYCLE_IDX] = min_values[self.CYCLE_IDX]
        elif self.Hardware_Arch == "DeFiNES":
            self.ref_point[self.ERROR_IDX] = min_values[self.ERROR_IDX]
            self.ref_point[self.EDP_IDX] = min_values[self.EDP_IDX]
        bd = DominatedPartitioning(ref_point=self.ref_point, Y=train_obj)
        volume = bd.compute_hypervolume().item()
        hvs.append(volume)
        # print()
        print("[init] Hypervolume: {}".format(self.N_BATCH, hvs[-1]))
        for iteration in range(1, self.N_BATCH + 1):
            self.iter = iteration
            fit_gpytorch_mll(mll)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.MC_SAMPLES])) # define the qEI and qNEI acquisition modules using a QMC sampler
            if self.acqu_algo == "Coflex":
                X = np.hstack(
                    (train_err_recording[:(len(train_err_recording) // 5) * 5].reshape(-1, 5), 
                     train_eng_recording[:(len(train_eng_recording) // 6) * 6].reshape(-1, 6))
                ) #(Dperf, fperf)
                y = torch.cat(
                    (torch.cat(self.observation_theta, dim=0).unsqueeze(1), 
                     torch.cat(self.observation_gamma, dim=0).unsqueeze(1)), 
                dim=1).numpy() # (Deng, feng)
                X_err = X[:, :5]
                X_eng = X[:, 5:]
                y_err = y[:, 0:1]
                y_eng = y[:, 1:2]
                fit_gpytorch_mll(mll_eng)
                fit_gpytorch_mll(mll_err)
                self.status = "ERR"
                theta_next, o_theta_next = self._get_new_data(self.image, self.acqu_algo, model_err, X_err, y_err, sampler) # θn+1, O(θn+1) -> ERR
                self.status = "ENG"
                gamma_next, o_gamma_next = self._get_new_data(self.image, self.acqu_algo, model_eng, X_eng, y_eng, sampler) # Γn+1, O(Γn+1) -> EDP
                self.cost_time = time.time() - self.start_time
                self.OPT_VS_TIME_REC = torch.cat([self.OPT_VS_TIME_REC, torch.cat((torch.cat((o_theta_next.unsqueeze(1), o_gamma_next.unsqueeze(1)), dim=1), torch.full((torch.cat((o_theta_next.unsqueeze(1), o_gamma_next.unsqueeze(1)), dim=1).shape[0], 1), self.cost_time, **tkwargs)), dim=1)])
                self.TOTAL_TRAIN_OBJ_REC = torch.cat([self.TOTAL_TRAIN_OBJ_REC, torch.cat((o_theta_next.unsqueeze(1), o_gamma_next.unsqueeze(1)), dim=1)])
                if theta_next == X_err and gamma_next == X_eng:
                    print("[INFO] Early Stop Happended")
                    break
                train_err_recording = np.append(train_err_recording, theta_next)
                train_eng_recording = np.append(train_eng_recording, gamma_next)
                self.observation_theta.append(o_theta_next) # O(θn+1)
                self.observation_gamma.append(o_gamma_next) # O(Γn+1)
                y_err = (torch.cat(self.observation_theta, dim=0)).numpy()
                y_eng = (torch.cat(self.observation_gamma, dim=0)).numpy()
                X = np.hstack(
                    (train_err_recording[:(len(train_err_recording) // 5) * 5].reshape(-1, 5), 
                     train_eng_recording[:(len(train_eng_recording) // 6) * 6].reshape(-1, 6))
                )
                y = np.stack((y_err, y_eng), axis=1)
                p_err, p_eng, p_x = self.pareto_check(iteration, X, y)
                p_x_array = np.array(p_x)
                self.beta = torch.tensor(p_x_array, **tkwargs)
                # ---------------------- Level2 --------------------- # 
                # print(self.beta.shape) # βn
                self.IntPar.append(self.beta) # DIntPar + βn
                On_beta = torch.stack((torch.tensor(p_err, **tkwargs), torch.tensor(p_eng, **tkwargs)), dim=1)# On(β)
                # print(On_beta.shape)
                assert self.beta.shape[1] == 11, "self.beta should have 11 columns"
                assert On_beta.shape[1] == 2, "On_beta should have 2 columns"
                observation_norm = (np.array(On_beta) - np.array(On_beta).min(axis=0)) / \
                            (np.array(On_beta).max(axis=0) - np.array(On_beta).min(axis=0) + 1e-8)            
                l1_norms = np.sum(observation_norm, axis=1)
                IntPar = self.beta[np.argsort(l1_norms)] # train_x
                self.DIntPar = On_beta # train_ob
                fit_gpytorch_mll(mll)
                # print("THIS **")
                self.status = "NOM"
                new_X_tensor, new_y_tensor = self._get_new_data(self.image, self.acqu_algo, model, IntPar, self.DIntPar, sampler) # βn+1, O(βn+1)
                # ***
                self.cost_time = time.time() - self.start_time
                self.OPT_VS_TIME_REC = torch.cat([self.OPT_VS_TIME_REC, torch.cat((new_y_tensor, torch.full((new_y_tensor.shape[0], 1), self.cost_time, **tkwargs)), dim=1)])
                self.TOTAL_TRAIN_OBJ_REC = torch.cat([self.TOTAL_TRAIN_OBJ_REC, new_y_tensor])
                # ***
                train_err_recording = np.append(train_err_recording, new_X_tensor[:, 0:5])
                train_eng_recording = np.append(train_eng_recording, new_X_tensor[:, 5:])
                self.observation_theta.append(new_y_tensor[:,0])
                self.observation_gamma.append(new_y_tensor[:,1])
                train_x = torch.tensor(
                    np.hstack(
                        (train_err_recording[:(len(train_err_recording) // 5) * 5].reshape(-1, 5), 
                         train_eng_recording[:(len(train_eng_recording) // 6) * 6].reshape(-1, 6))
                    )
                )
                train_obj = torch.cat(
                    (torch.cat(self.observation_theta, dim=0).unsqueeze(1), 
                     torch.cat(self.observation_gamma, dim=0).unsqueeze(1)), 
                dim=1)
                p_err, p_eng, p_x = self.pareto_check(iteration, train_x.numpy(), train_obj.numpy())
                p_x_array = np.array(p_x)
                train_x = torch.tensor(p_x_array, **tkwargs)
                train_obj = torch.stack((torch.tensor(p_err, **tkwargs), torch.tensor(p_eng, **tkwargs)), dim=1)
                assert train_x.shape[1] == 11, "train_x should have 11 columns"
                assert train_obj.shape[1] == 2, "train_obj should have 2 columns"
                assert train_x.shape[0] == train_obj.shape[0], "train_x and train_obj should have the same number of rows"
                assert train_x.ndim == 2, "train_x should be a 2D tensor"
                assert train_obj.ndim == 2, "train_obj should be a 2D tensor"
                mll, model = self._initialize_model(train_x, train_obj, self.bounds_fornorm)
                mll_err, model_err = self._initialize_model(train_x[:, 0:5], train_obj[:, 0].unsqueeze(1), self.bounds_err)
                mll_eng, model_eng = self._initialize_model(train_x[:, 5:], train_obj[:, 1].unsqueeze(1), self.bounds_eng)
                bd = DominatedPartitioning(ref_point=self.ref_point, Y=self.TOTAL_TRAIN_OBJ_REC)
                volume = bd.compute_hypervolume().item()
                hvs.append(volume)
            else:
                new_x, new_obj = self._get_new_data(self.image, self.acqu_algo, model, train_x, train_obj, sampler)
                train_x = torch.cat([train_x, new_x])
                train_obj = torch.cat([train_obj, new_obj])
                bd = DominatedPartitioning(ref_point=self.ref_point, Y=train_obj)
                volume = bd.compute_hypervolume().item()
                hvs.append(volume)
            if not (self.acqu_algo == "random" or self.acqu_algo == "Coflex"):
                mll, model = self._initialize_model(train_x, train_obj, self.bounds_fornorm)
            print("iteration [{}/{}] Hypervolume: {}".format(iteration, self.N_BATCH, hvs[-1]))
        # Show top-5 result
        # self.delete_all_folders('inputs/WL/Meta_prototype_DF')
        print('+++++++++++ Result +++++++++++')
        torch.set_printoptions(precision=2, linewidth=100)
        print("H-Volume: ", hvs[-1])
        if self.Hardware_Arch == "DL2":
            if self.benchmark == "sss":
                optimal_x = train_x[-1-self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1-self.BATCH_SIZE:-1]
                print("Backbone Design Space: NATS-Bench-SSS")
                for i in range(self.BATCH_SIZE):
                    arch = '{}:{}:{}:{}:{}'.format(int(optimal_x[i,0]),int(optimal_x[i,1]),int(optimal_x[i,2]),int(optimal_x[i,3]),int(optimal_x[i,4]))
                    acce = {"X1":optimal_x[i,5].item(), "X2":optimal_x[i,6].item(), "X3":optimal_x[i,7].item(), "X4":optimal_x[i,8].item(), "X5":optimal_x[i,9].item(), "X6":optimal_x[i,10].item()}
                    print("* Candidate[{}]".format(i+1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDL2 Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tRBFleX Score: ", optimal_obj[i,0].item())
                    print("\tEnergy (FAKE): ", -1*optimal_obj[i,1].item())
                    print("\tCycle Count (FAKE): ", int(-1*optimal_obj[i,2].item()))
            elif self.benchmark == "201":
                optimal_x = train_x[-1-self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1-self.BATCH_SIZE:-1]
                print("Backbone Design Space: NAS-Bench-201")
                for i in range(self.BATCH_SIZE):
                    action = list(map(int, optimal_x[i, 0:6]))
                    arch = self.policy.generate_arch(action)
                    acce = {"X1":optimal_x[i,5].item(), "X2":optimal_x[i,6].item(), "X3":optimal_x[i,7].item(), "X4":optimal_x[i,8].item(), "X5":optimal_x[i,9].item(), "X6":optimal_x[i,10].item()}
                    print("* Candidate[{}]".format(i+1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDL2 Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tRBFleX Score: ", optimal_obj[i,0].item())
                    print("\tEnergy (FAKE): ", -1*optimal_obj[i,1].item())
                    print("\tCycle Count (FAKE): ", int(-1*optimal_obj[i,2].item()))
        elif self.Hardware_Arch == "ScaleSim":
            if self.benchmark == "sss":
                optimal_x = train_x[-1-self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1-self.BATCH_SIZE:-1]
                print("Backbone Design Space: NATS-Bench-SSS")
                for i in range(self.BATCH_SIZE):
                    arch = '{}:{}:{}:{}:{}'.format(int(optimal_x[i,0]),int(optimal_x[i,1]),int(optimal_x[i,2]),int(optimal_x[i,3]),int(optimal_x[i,4]))
                    acce = {"X1":optimal_x[i,5].item(), "X2":optimal_x[i,6].item(), "X3":optimal_x[i,7].item(), "X4":optimal_x[i,8].item(), "X5":optimal_x[i,9].item(), "X6":optimal_x[i,10].item()}
                    print("* Candidate[{}]".format(i+1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tHardware: ", acce)
                    print("\t----------------------------------")
                    print("\tRBFleX Score: ", optimal_obj[i,0].item())
                    print("\tCycle Count (FAKE): ", int(-1*optimal_obj[i,1].item()))
            elif self.benchmark == "201":
                optimal_x = train_x[-1-self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1-self.BATCH_SIZE:-1]
                print("Backbone Design Space: NAS-Bench-201")
                for i in range(self.BATCH_SIZE):
                    action = list(map(int, optimal_x[i, 0:6]))
                    arch = self.policy.generate_arch(action)
                    acce = {"X1":optimal_x[i,5].item(), "X2":optimal_x[i,6].item(), "X3":optimal_x[i,7].item(), "X4":optimal_x[i,8].item(), "X5":optimal_x[i,9].item(), "X6":optimal_x[i,10].item()}
                    print("* Candidate[{}]".format(i+1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDL2 Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tRBFleX Score: ", optimal_obj[i,0].item())
                    print("\tCycle Count (FAKE): ", int(-1*optimal_obj[i,1].item()))
        elif self.Hardware_Arch == "DeFiNES":
            if self.benchmark == "sss":
                optimal_x = train_x
                optimal_obj = train_obj
                print("Backbone Design Space: NATS-Bench-SSS")
                for i in range(len(optimal_x)):
                    arch = '{}:{}:{}:{}:{}'.format(int(optimal_x[i, 0]), int(optimal_x[i, 1]), int(optimal_x[i, 2]),
                                                   int(optimal_x[i, 3]), int(optimal_x[i, 4]))
                    acce = {"X1": optimal_x[i, 5].item(), "X2": optimal_x[i, 6].item(), "X3": optimal_x[i, 7].item(),
                            "X4": optimal_x[i, 8].item(), "X5": optimal_x[i, 9].item(), "X6": optimal_x[i, 10].item()}
                    print("* Candidate[{}]".format(i + 1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDeFiNES Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tErr ", (optimal_obj[i, 0].item()), "%")
                    print("\tEDP: ",(optimal_obj[i, 1].item()), "µJ*s")
            elif self.benchmark == "201":
                optimal_x = train_x[-1-self.BATCH_SIZE:-1]
                optimal_obj = train_obj[-1-self.BATCH_SIZE:-1]
                print("Backbone Design Space: NATS-Bench-201")
                for i in range(len(optimal_x)):
                    action = list(map(int, optimal_x[i, 0:6]))
                    arch = self.policy.generate_arch(action)
                    acce = {"X1":optimal_x[i,5].item(), "X2":optimal_x[i,6].item(), "X3":optimal_x[i,7].item(), "X4":optimal_x[i,8].item(), "X5":optimal_x[i,9].item(), "X6":optimal_x[i,10].item()}
                    print("* Candidate[{}]".format(i+1))
                    print("\tBackbone architecture: {}".format(arch))
                    print("\tDeFiNES Hardware: ", acce)
                    print("\t----------------------------------")
                    print("\tErr ", optimal_obj[i,0].item(), "%")
                    print("\tEDP: ",(-1*optimal_obj[i,1].item()), "µJ*s")
        base_path = 'COFleX_result/' + self.Hardware_Arch + '_SSS_' + str(self.mapping) + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_")
        os.makedirs(base_path, exist_ok=True)
        save_path = 'COFleX_result/' + self.Hardware_Arch + '_SSS_' + str(self.mapping) + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/hvs.csv'
        np.savetxt(save_path, hvs)
        save_path = 'COFleX_result/' + self.Hardware_Arch + '_SSS_' + str(self.mapping) + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/train_output.csv'
        np.savetxt(save_path, train_obj.cpu().numpy())
        save_path = 'COFleX_result/' + self.Hardware_Arch + '_SSS_' + str(self.mapping) + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/train_input.csv'
        np.savetxt(save_path, train_x.cpu().numpy())
        save_path = 'COFleX_result/' + self.Hardware_Arch + '_SSS_' + str(self.mapping) + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/train_input.csv'
        np.savetxt(save_path, train_x.cpu().numpy())
        save_path = 'COFleX_result/' + self.Hardware_Arch + '_SSS_' + str(self.mapping) + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/opt_efficiency_analys.csv'
        np.savetxt(save_path, self.search_time_rec.cpu().numpy())
        visit_time = torch.tensor([[self.estm_vist_time, self.TOTAL_RUN_TIME]], **tkwargs)
        save_path = 'COFleX_result/' + self.Hardware_Arch + '_SSS_' + str(self.mapping) + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/comput_cost_analys.csv'
        np.savetxt(save_path, visit_time.cpu().numpy())
        save_path = 'COFleX_result/' + self.Hardware_Arch + '_SSS_' + str(self.mapping) + '_ALGO_' +self.set_acqu_algo + '_TIME_' + datetime.now().strftime("_%m_%d_%H_%M_") + '/opt_vs_time_analys.csv'
        np.savetxt(save_path, self.OPT_VS_TIME_REC.cpu().numpy())
        self.plots(hvs, train_obj)
        return train_obj, train_x
    def pareto_check(self, iteration, X, y):
        P = []  
        F = defaultdict(list)  
        for i in range(X.shape[0]):
            P.append(Individual())
            P[i].solution = X[i] 
            P[i].calculate_objective(y[i])  
        F = self.fast_non_dominated_sort(P)
        for i in range(1, len(F) + 1):
            if not F[i]:  # Check if L is an empty list
                # print("_SKIP_")
                continue
            self.crowding_distance_assignment(F[i])  
            F[i].sort(key=lambda x: x.distance)  
        P = []
        for i in range(min(self.cluster_lim, len(F))):
            if F[i]:
                t = []
                t.append(F[i])
                P.extend(t)
                # plt.clf()
        plt.title('current generation:' + str(iteration))
        self.plot_P(P)
        for t in (P):  
            p_err = [ind.objective[0] for ind in t]
            p_eng = [ind.objective[1] for ind in t]
            p_x = [ind.solution for ind in t]
        return p_err,p_eng,p_x
    def acquisition_function(self, gp, x, y_min):
        mu, sigma = gp.predict(x, return_std=True)
        sigma = np.clip(sigma, 1e-9, None)
        z = (mu - y_min) / sigma
        ei = (mu - y_min) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei
    def optimize_acquisition(self, gp, X, bounds, y_min): # the best performance y and random chose x
        def objective(X):
            X = X.reshape(1, -1)
            return self.acquisition_function(gp, X, y_min)
        idx = random.randint(0, len(X)-1)
        result = minimize(objective, 
                        x0=X[idx], # initial generate candidate
                        bounds=bounds)
        return result.x
class Individual(object):
    def __init__(self):
        self.solution = None  
        self.objective = defaultdict()
        self.n = 0  
        self.rank = 0  
        self.S = []  
        self.distance = 0  
    def bound_process(self, bound_min, bound_max):
        for i, item in enumerate(self.solution):
            if item > bound_max:
                self.solution[i] = bound_max
            elif item < bound_min:
                self.solution[i] = bound_min
    def calculate_objective(self, objective_fun):
        self.objective = objective_fun
    def __lt__(self, other):
        v1 = self.objective  
        v2 = other.objective  
        for i in range(len(v1)):
            if v1[i] > v2[i]:  
                return 0  
        return 1