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
Coflex is a hardware-aware neural architecture search (HW-NAS) optimizer that jointly considers key parameters from the software-side neural network architecture and corresponding hardware design configurations. It operates through an iterative co-optimization framework consisting of a multi-objective Bayesian optimizer (front-end) and a performance evaluator (back-end).

In each optimization iteration, Coflex takes candidate configurations as input, evaluates their actual performance trade-offs between software accuracy (e.g., error rate) and hardware efficiency (e.g., energy-delay product), and updates the surrogate models in the Bayesian optimizer accordingly. This process enables Coflex to progressively refine the Pareto front toward a designated reference point (e.g., (0,0)) in the objective space, effectively navigating the inherent conflict between software and hardware objectives.

After multiple iterations, Coflex converges to a near-globally optimal Pareto front, where each point represents a non-dominated configuration offering an optimal trade-off between software performance and hardware cost. The final output provides interpretable architectural design recommendations for both neural network developers and hardware architects, along with the expected performance metrics of each configuration. As a result, Coflex delivers an automated, end-to-end software-hardware co-design pipeline.

<p align="center"><img width=100% src="https://github.com/myh1998/Coflex/blob/main/Figs/Fig_hw_framework_overview.png"></p>

### ◼️ Search Space Define
The search space of HW-NAS encompasses a high-dimensional hyperparameter space composed of both software-wise parameters (e.g., neural network architectural choices) and hardware-wise parameters (e.g., hardware resource configurations). To initialize the optimization process, Coflex performs uniform sampling across all dimensions of this joint search space. These sampled configurations are then used to construct the initial Gaussian surrogate models within the multi-objective Bayesian optimization front-end.

<p align="center"><img width=30% src="https://github.com/myh1998/Coflex/blob/main/Figs/Fig_search_space.png"></p>

### ◼️ Total Hyper-parameters for different NAS-Benchmark suites
This work leverages multiple standardized NAS benchmark suites to provide consistent neural architecture input representations for the Coflex optimizer. These benchmarks serve as the input source for both software and hardware configuration spaces.

If you wish to run Coflex on a specific NAS benchmark, please refer to the table below for the corresponding repository links. Make sure to download and store the datasets according to the instructions provided in the [How to Run](https://github.com/myh1998/Coflex/blob/main/README.md#-how-to-run) section.

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
Coflex tackles the scalability bottlenecks in hardware-aware NAS by introducing a two-level sparse Gaussian process (SGP) framework:

🔹 Per-objective SGPs reduce complexity by modeling each optimization objective separately.

🔹 Pareto-based fusion combines these models using non-dominance filtering to preserve multi-objective structure.

This design enables Coflex to efficiently explore massive software-hardware search spaces (10¹⁹+ configs) while maintaining high-fidelity trade-off modeling.

<p align="center"><img width=80% src="https://github.com/myh1998/Coflex/blob/main/Figs/Fig_dimension_decomposation.png"></p>

### ◼️ Sparse Gaussian inducing strategies
To handle the scalability bottlenecks of standard Gaussian Processes in large-scale HW-NAS tasks, Coflex adopts sparse GP modeling with inducing points. Instead of maintaining a full covariance matrix, Coflex approximates it using a low-rank structure derived from a small set of representative inducing inputs. This significantly reduces computational cost and improves stability, enabling fast and reliable optimization over high-dimensional software-hardware design spaces.

## 🟨 Repository File Structure
### ◼️ Multiple Bayesian Optimizer(Front-end)
🔹 [FRCN_Simulator](https://github.com/myh1998/Coflex/blob/main/Simulator/FRCN_Simulator.py)

### ◼️ Performance Evaluator(Back-end)
#### 🧠 Network Evaluator
🔹[RBFleX-NAS](https://github.com/myh1998/Coflex/blob/main/Simulator/RBFleX.py)

#### ⚙️ Hardware Evaluator
Please download the hardware deployment analyzer from the following link and follow the instructions in [Preprocessing for Reproduction](https://github.com/myh1998/Coflex/blob/main/README.md#preprocessing-for-reproduction) section to correctly install it for reproducing the results presented in the paper.

🔹[DeFiNES](https://sutdapac-my.sharepoint.com/:f:/g/personal/yinhui_ma_mymail_sutd_edu_sg/EhqUH-LOmt5PmVbKjIocAUUBLzoJ0s_6Y2oSfbvpmvkh1g?e=uM1249)

🔹[Scale-Sim](https://sutdapac-my.sharepoint.com/:f:/g/personal/yinhui_ma_mymail_sutd_edu_sg/EtNUdprB7QVEvcZk54zrEXMBN0tAR-iSGE1J-f0utFUxVw?e=GsgJ7s)

## 🟨 Installation Requirements
```python
pip install -r requirements.txt
```
[Requirements](https://github.com/myh1998/Coflex/blob/main/requirements.txt)

## 🟨 How to Run
### ◼️ Preprocessing for Reproduction
Please follow the steps below to correctly set up the working environment for reproducing the experimental results of COFleX:

🔹Set the Working Directory
  Choose 
  ```python
  COFleX/
  ```
  as the root working directory.

🔹Unpack Required Archives
  ```python
  unzip COFleX_Analysis.zip -d COFleX/

  unzip design_space.zip -d COFleX/
  ```
🔹Prepare Dataset
  Download the ImageNet val dataset and place it into the following directory:
  ```python
  COFleX/dataset/
  ```
🔹Install Required Simulators
  Download and place the DeFiNES & Scale-Sim into the specified directory:
  ```python
  COFleX/Simulator/
  ```
  Please ensure all environment variables and simulator dependencies are properly configured as described in each simulator's official documentation.

### ◼️ Reproduce the results in Workload1 (Global Search in NATS Benchmark)
```python
# Global Search in NATS Benchmark
# Supported Datasets: CIFAR10, CFIAR100, ImageNet
# Executed task: Image Classification
python run_sss.py

```


