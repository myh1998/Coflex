# LAXOR: A BNN Accelerator with Latch-XOR Logic for Local Computing

## 🟨 Contents
- [Introduction](https://github.com/tomomasayamasaki/LAXOR#-introduction)
    - [LAXOR Accelerator](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-laxor-accelerator)
    - [LAXOR Accelerator Simulator](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-laxor-accelerator-simulator)
- [Repogitory File Structure](https://github.com/tomomasayamasaki/LAXOR#-repository-file-structure)
    - [LAXOR_Sim](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-laxor_sim)
    - [Program and pre-trained model for test running](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-program-and-pre-trained-model-for-test-running)
- [Installation Requirements](https://github.com/tomomasayamasaki/LAXOR#-installation-requirements)
- [How to Run](https://github.com/tomomasayamasaki/LAXOR#-how-to-run)
    - [Areca platform](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-areca-platform)
    - [Load pre-trained model](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-load-pre-trained-model)
    - [Config](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-config)
    - [Run Example](https://github.com/tomomasayamasaki/LAXOR#%EF%B8%8F-run-example)
- [Citing LAXOR accelerator and simulator](https://github.com/tomomasayamasaki/LAXOR#-citing-laxor-accelerator-and-simulator)
- [Licence](https://github.com/tomomasayamasaki/LAXOR#-licence)

## 🟨 Introduction
### ◼️ Coflex optimizer framework
Coflex is a hardware-aware neural architecture search (HW-NAS) optimizer that jointly considers key parameters from the software-side neural network architecture and corresponding hardware design configurations. It operates through an iterative co-optimization framework consisting of a multi-objective Bayesian optimizer (front-end) and a performance evaluator (back-end).

In each optimization iteration, Coflex takes candidate configurations as input, evaluates their actual performance trade-offs between software accuracy (e.g., error rate) and hardware efficiency (e.g., energy-delay product), and updates the surrogate models in the Bayesian optimizer accordingly. This process enables Coflex to progressively refine the Pareto front toward a designated reference point (e.g., (0,0)) in the objective space, effectively navigating the inherent conflict between software and hardware objectives.

After multiple iterations, Coflex converges to a near-globally optimal Pareto front, where each point represents a non-dominated configuration offering an optimal trade-off between software performance and hardware cost. The final output provides interpretable architectural design recommendations for both neural network developers and hardware architects, along with the expected performance metrics of each configuration. As a result, Coflex delivers an automated, end-to-end software-hardware co-design pipeline.

<p align="center"><img width=60% src="https://github.com/tomomasayamasaki/LAXOR/blob/main/README/LAXOR_Core.png"></p>

### ◼️ Search Space Define
#### xxx_xxx

### ◼️ Total Hyper-parameters for different NAS-Benchmark suites
#### xxx_xxx

### ◼️ Dimension Decomposition
#### xxx_xxx

### ◼️ Sparse Gaussian inducing strategies
#### xxx_xxx

## 🟨 Repository File Structure

## 🟨 Installation Requirements

## 🟨 How to Run
