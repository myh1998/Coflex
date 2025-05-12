# Coflex: Enhancing HW-NAS with Sparse Gaussian Processes for Efficient and Scalable Software-Hardware Co-Design

## 🟨 Contents
- [Introduction](https://github.com/myh1998/Coflex#-introduction)
  - [Coflex Optimizer Framework](https://github.com/myh1998/Coflex#%EF%B8%8F-coflex-optimizer-framework)
  - [Search Space Define](https://github.com/myh1998/Coflex#%EF%B8%8F-search-space-define)
  - [Total Hyper-Parameters For Different NAS Benchmark Suites](https://github.com/myh1998/Coflex#%EF%B8%8F-total-hyper-parameters-for-different-nas-benchmark-suites)
  - [Dimension Decomposition](https://github.com/myh1998/Coflex#%EF%B8%8F-dimension-decomposition)
  - [Sparse Gaussian Inducing Strategies](https://github.com/myh1998/Coflex#%EF%B8%8F-sparse-gaussian-inducing-strategies)
- [Repository File Structure](https://github.com/myh1998/Coflex#-repository-file-structure)
  - [Multiple Bayesian Optimizer](https://github.com/myh1998/Coflex#%EF%B8%8F-multiple-bayesian-optimizerfront-end)
  - [Performance Evaluator](https://github.com/myh1998/Coflex#%EF%B8%8F-performance-evaluatorback-end)
    - [Network Evaluator - RBFleX-NAS](https://github.com/myh1998/Coflex#-network-evaluator)
    - [Hardware Evaluator - DeFiNES & Scale-Sim](https://github.com/myh1998/Coflex#%EF%B8%8F-hardware-evaluator)
- [Installation Requirements](https://github.com/myh1998/Coflex#-installation-requirements)
- [How To Run](https://github.com/myh1998/Coflex#-how-to-run)
  - [Preprocessing For Reproduction](https://github.com/myh1998/Coflex#%EF%B8%8F-preprocessing-for-reproduction)
  - [Reproduce The Results In Workload1](https://github.com/myh1998/Coflex#%EF%B8%8F-reproduce-the-results-in-workload1)  

## 🟨 Introduction
### ◼️ Coflex optimizer framework
`Coflex` is a hardware-aware neural architecture search (HW-NAS) optimizer that jointly considers key parameters from the software-side neural network architecture and corresponding hardware design configurations. It operates through an iterative co-optimization framework consisting of a multi-objective Bayesian optimizer (front-end) and a performance evaluator (back-end).

In each optimization iteration, Coflex takes candidate configurations as input, evaluates their actual performance trade-offs between software accuracy (e.g., error rate) and hardware efficiency (e.g., energy-delay product), and updates the surrogate models in the Bayesian optimizer accordingly. This process enables Coflex to progressively refine the Pareto front toward a designated reference point (e.g., (0,0)) in the objective space, effectively navigating the inherent conflict between software and hardware objectives.

After multiple iterations, Coflex converges to a near-globally optimal Pareto front, where each point represents a non-dominated configuration offering an optimal trade-off between software performance and hardware cost. The final output provides interpretable architectural design recommendations for both neural network developers and hardware architects, along with the expected performance metrics of each configuration. As a result, Coflex delivers an automated, end-to-end software-hardware co-design pipeline.

<p align="center"><img width=70% src="https://github.com/myh1998/Coflex/blob/main/Figs/Move/Fig_hw_framework_overview_v2_page-0001.jpg""></p>

### ◼️ Search Space Define
The `search space` of HW-NAS encompasses a high-dimensional hyperparameter space composed of both software-wise parameters (e.g., neural network architectural choices) and hardware-wise parameters (e.g., hardware resource configurations). To initialize the optimization process, Coflex performs uniform sampling across all dimensions of this joint search space. These sampled configurations are then used to construct the initial Gaussian surrogate models within the multi-objective Bayesian optimization front-end.

<p align="center"><img width=40% src="https://github.com/myh1998/Coflex/blob/main/Figs/Move/Fig_design_space_of_nas_sss_page-0001.jpg""></p>

### ◼️ Total Hyper-parameters for different NAS-Benchmark suites
This work leverages multiple standardized NAS benchmark suites to provide consistent neural architecture input representations for the Coflex optimizer. These benchmarks serve as the input source for both software and hardware configuration spaces.

If you wish to run Coflex on a specific NAS benchmark, please refer to the table below for the corresponding repository links. Make sure to download and store the datasets according to the instructions provided in the [`How to Run`](https://github.com/myh1998/Coflex/blob/main/README.md#-how-to-run) section.

Coflex is designed with high extensibility, supporting diverse NAS benchmarks across various tasks. If you intend to apply Coflex to a new benchmark not covered in this work, you may edit the internal data mapping logic in the Software Performance Evaluator and Hardware Performance Evaluator (DeFiNES) modules to ensure compatibility with the new input/output format.

> **Note**:  
> - **Hw space** = Hardware search space size  
> - **Sw space** = Software search space size  
> - **Total Parameters** = Joint search space size = Hw × Sw

| **Suite**                | **NATS-Bench-SSS** | **TransNAS-Bench-101** | **NAS-Bench-201** | **NAS-Bench-NLP** |
|--------------------------|--------------------|-------------------------|-------------------|-------------------|
| ⚙️ **Hw space**          | 2.81×10¹⁴          | 2.81×10¹⁴              | 2.81×10¹⁴         | 2.02×10¹⁵         |
| 🧠 **Sw space**          | 3.20×10⁴           | 4.10×10³               | 6.50×10³          | 1.43×10⁴          |
| 📈 **Total Parameters**  | 9.22×10¹⁸          | 1.15×10¹⁸              | 1.83×10¹⁸         | 2.89×10¹⁹         |


### ◼️ Dimension Decomposition
Coflex tackles the scalability bottlenecks in hardware-aware NAS by introducing a `two-level sparse Gaussian process (SGP)` framework:

🔹 Per-objective SGPs reduce complexity by modeling each optimization objective separately.

🔹 Pareto-based fusion combines these models using non-dominance filtering to preserve multi-objective structure.

This design enables Coflex to efficiently explore massive software-hardware search spaces (10¹⁹+ configs) while maintaining high-fidelity trade-off modeling.

<p align="center"><img width=80% src="https://github.com/myh1998/Coflex/blob/main/Figs/Move/Fig_dimension_decomposation_page-0001.jpg"></p>

### ◼️ Sparse Gaussian inducing strategies
To handle the scalability bottlenecks of standard Gaussian Processes in large-scale HW-NAS tasks, Coflex adopts sparse GP modeling with inducing points. Instead of maintaining a full covariance matrix, Coflex approximates it using a low-rank structure derived from a small set of representative inducing inputs. This significantly reduces computational cost and improves stability, enabling fast and reliable optimization over high-dimensional software-hardware design spaces.

## 🟨 Repository File Structure
### ◼️ Multiple Bayesian Optimizer(Front-end)
🔹Download Link: [`FRCN_Simulator`](https://github.com/myh1998/Coflex/blob/main/Simulator/FRCN_Simulator.py)

### ◼️ Performance Evaluator(Back-end)
#### 🧠 Network Evaluator
🔹Download Link: [`RBFleX-NAS`](https://github.com/myh1998/Coflex/blob/main/Simulator/RBFleX.py)

#### ⚙️ Hardware Evaluator

This project supports two types of hardware deployment evaluators: `DeFiNES` and `Scale-Sim`, each offering distinct trade-offs between evaluation speed and accuracy:
```bash
# Scale-Sim is employed as a fast yet lower-accuracy evaluator.
# Average evaluation time: 3–5 seconds per query
# Output: Estimated cycle count
# Use case: Suitable for quick, large-scale architecture assessments during the early-stage search or pruning processes.

# DeFiNES serves as a high-accuracy, hardware-faithful evaluator, albeit with slower evaluation speed.
# Average evaluation time: ~200 seconds per query
# Accuracy:
#  Average latency prediction error: ~3%
#  Worst-case latency error (e.g., FSRCNN): up to 10%
#  Energy prediction error: within 6%
# Use case: Ideal for precise, end-stage performance estimation and final candidate ranking.
```
Please download the hardware deployment evaluator from the following link and follow the instructions in [`Preprocessing for Reproduction`](https://github.com/myh1998/Coflex/blob/main/README.md#preprocessing-for-reproduction) section to correctly install it for reproducing the results presented in the paper.

🔹Download Link: [`DeFiNES`](https://sutdapac-my.sharepoint.com/:f:/g/personal/yinhui_ma_mymail_sutd_edu_sg/EhqUH-LOmt5PmVbKjIocAUUBLzoJ0s_6Y2oSfbvpmvkh1g?e=uM1249)

🔹Download Link: [`Scale-Sim`](https://sutdapac-my.sharepoint.com/:f:/g/personal/yinhui_ma_mymail_sutd_edu_sg/EtNUdprB7QVEvcZk54zrEXMBN0tAR-iSGE1J-f0utFUxVw?e=GsgJ7s)

## 🟨 Installation Requirements
```bash
pip install -r requirements.txt
```
[`Requirements`](https://github.com/myh1998/Coflex/blob/main/requirements.txt)

## 🟨 How to Run
### ◼️ Preprocessing for Reproduction
Please follow the steps below to correctly set up the working environment for reproducing the experimental results of COFleX:

🔹Set the Working Directory
  Choose 
  ```bash
  cd COFleX/
  ```
  as the root working directory.

🔹Unpack Required Archives
  ```bash
  unzip COFleX_Analysis.zip -d COFleX/
  unzip design_space.zip -d COFleX/
  ```

🔹Download & Unzip NAS-Benchmark
  > The Coflex framework supports multiple NAS benchmarks. Please use the corresponding download links as needed.
  
  > For NATS-Bench-SSS, Download Link: [`NATS-sss-v1_0-50262-simple`](https://onedrive.live.com/?authkey=%21AKSvuIkSXx0UQaI&id=8305A36BB9DB1CA9%21127&cid=8305A36BB9DB1CA9)
  ```bash
  unzip NATS-sss-v1_0-50262-simple.zip -d COFleX/
  ```

🔹Prepare Dataset
  Download the ImageNet/val dataset and place it into the following directory:
  > The `CIFAR-10` and `CIFAR-100` datasets will be **automatically downloaded** by the program into `COFleX/dataset/`.  
  > The `ImageNet/val` subset must be **manually downloaded** or obtained via the command line if a valid URL is available:
  >
  ```bash
   wget "https://your-server.com/path-to/imagenet_val.zip" -O imagenet_val.zip
   mkdir -p COFleX/dataset/
   unzip imagenet_val.zip -d COFleX/dataset/val/
  ```

🔹Install Required Simulators
  Download and place the DeFiNES & Scale-Sim into the specified directory:
  ```bash
  unzip DeFiNES.zip -d COFleX/Simulator/
  unzip ScaleSim.zip -d COFleX/Simulator/
  ```
  Please ensure all environment variables and simulator dependencies are properly configured as described in each simulator's official documentation.

### ◼️ Reproduce the results in Workload1 (Global Search in NATS Benchmark)
> This work supports diverse workload inputs. Please refer to the following section for parameter redefinitions to adapt the implementation to your local execution environment:
```python
# run_sss.py
  # * Line 5
    acc_code_path = "your-path-to/COFleX/COFleX_Analysis/RBFleX/imageNet_SSS"
  
  # * Line 108 ~ 113
    for N_HYPER in [10]: # 5,10,30
      for ACQU in ["Coflex","qNParEGO","qNEHVI","qEHVI","random,"nsga", "pabo"]: # "Coflex","qNParEGO","qNEHVI","qEHVI","random,"nsga", "pabo" 
          for ITERS in [30]: # 5, 15, 30, 45
              for N_INIT in [100]: # 10,50,100,300
                  for BS in [10]: # 1,4,10
                      for H_ARCH in ["DeFiNES"]: # "ScaleSim", "DeFiNES"
  
  # * Line 182 & 183
    parser.add_argument('-ih','--IN-H', default='your-image-H_szie', type=int, help='Height of input image for faster RCNN (default: 224)') # 224, 32
    parser.add_argument('-iw','--IN-W', default='your-image-W_szie', type=int, help='Width of input image for faster RCNN (default: 224)') # 224, 32

# Simulator/FRCN_Simulator.py
  # * Line 113
    benchmark_root="your-path-to/COFleX/NATS-sss-v1_0-50262-simple",
  # * Line 144
    img_root="your-path-to/COFleX/COFleX/dataset"
# COFleX_Analysis/RBFleX/imageNet_SSS/Check_acc.py
  # * Line 16
    api_loc = 'your-path-to/COFleX/NATS-sss-v1_0-50262-simple'
  # * Line 20
    accuracy, latency, time_cost, current_total_time_cost = searchspace.simulate_train_eval(uid, dataset='select-dataset-as-you-want!', hp='90') # "cifar10", "cifar100" "ImageNet16-120"
```

> To reproduce the Figs/Tabs results, simply start with `run_sss.py`
```python
# Global Search in NATS Benchmark
# Supported Datasets: CIFAR10, CFIAR100, ImageNet
# Executed task: Image Classification
python run_sss.py
```

> Output Results Storage Location & Figs Reproduce
> When the program completes execution successfully, the results will be stored under `COFleX\COFleX_result\`, which will include:
```bash
# train_input.py, representing the final software and hardware parameters generated through the HW-NAS
# optimization process  

# train_output.py, representing the results obtained in each objective dimension during multi-objective
# optimization, which form the Pareto front  

# hv.py, containing the Dominated Hypervolume progression of all solution sets searched by each HW-NAS
# method in every iteration  

# opt_vs_time_analys.py, recording the solutions retained by each HW-NAS method during each iteration,
# demonstrating the optimization efficiency and convergence ability over time  

# opt_efficiency_analys, recording the maximized software performance and minimized hardware consumption
#achieved in each dimension during iterative optimization  
```

> To easily reproduce the figures presented in the paper, you may optionally download from[`Results Saving`](https://sutdapac-my.sharepoint.com/:f:/g/personal/yinhui_ma_mymail_sutd_edu_sg/EmZNWvydDENCv9hQXs6U4aMBsfzEkL_HztQJUX91KgTadw?e=yeS7QL)

> The [`saving`](https://github.com/myh1998/Coflex/tree/main/saving) folder contains five figure plotting scripts:
```bash
# 1_run_ploting_pareto_fronts.py, used to plot the Pareto front formed by multi-objective optimization,
illustrating the trade-off relationships  

# 2_run_inverted_generational_dis.py, used to show the Pareto front closest to the reference point (0, 0).
The hyper-space enclosed by this front and the reference point is called the Pareto Optimal Region,
demonstrating the algorithm’s contraction and advancement capability. The smaller the value, the better
the final optimized solution set   

# 3_run_hypervolume.py, used to show the Dominated Hypervolume of all solution sets searched by the HW-NAS
algorithm over multiple iterations, reflecting the algorithm’s exploration ability in the search space.
A larger value indicates a more comprehensive exploration, avoiding local optima  

# 4_run_opt_efficiency_analysis.py, records the solutions retained by each HW-NAS method during
each iteration, demonstrating optimization efficiency and convergence ability across iterations  

# 5_run_opt_vs_time_analysis.py, records the maximized software performance and minimized hardware
consumption achieved in each dimension during iterative optimization  
```
> Please refer to the following section for `your-path-to` redefinitions to adapt the implementation to
> your local execution environment, then you may run:
```python
python 1_run_ploting_pareto_fronts.py
```
> To reproduce the results presented in Figure 4(a) of the paper. The expected output is illustrated as follow.

<p align="center"><img width=50% src="https://github.com/myh1998/Coflex/blob/main/Figs/Move/Fig_pareto_fronts_page-0001.jpg"></p>

```python
python 2_run_inverted_generational_dis.py
```
> To reproduce the results presented in Figure 4(b) of the paper. The expected output is illustrated as follow.

<p align="center"><img width=50% src="https://github.com/myh1998/Coflex/blob/main/Figs/Move/Fig_pareto_region_page-0001.jpg"></p>

```python
python 3_run_hypervolume.py
```
> To reproduce the results presented in Figure 4(c) of the paper. The expected output is illustrated as follow.

<p align="center"><img width=50% src="https://github.com/myh1998/Coflex/blob/main/Figs/Move/Fig_hv_page-0001.jpg"></p>

```python
python 4_run_opt_efficiency_analysis.py
```
> To reproduce the results presented in Figure 4(d)&(e)&(f) of the paper. The expected output is illustrated as follow.

<p align="center"><img width=85% src="https://github.com/myh1998/Coflex/blob/main/Figs/Move/comb-figs.jpg"></p>

```python
python 5_run_opt_vs_time_analysis.py 
```
> To reproduce the results presented in Figure 4(f) of the paper. The expected output is illustrated as follow. For coflex, its optimization process demonstrates better stability, maintaining a lower Err vs EDP relationship in both the early and later stages, with a clear convergence appearing within the limited number of iterations, indicating that coflex may possess global optimal search capabilities. Compared to other methods, coflex has a better optimization advantage.

<p align="center"><img width=90% src="https://github.com/myh1998/Coflex/blob/main/Figs/Fig_opt_time_analysis.png"></p>

> If you wish to retrain all HW-NAS algorithms on different workloads, please copy the result package from `COFleX\COFleX_result\` into the `Results Saving` directory, and update the `path` configs in all scripts under the `saving` folder to match your local deployment environment.

