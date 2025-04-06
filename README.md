# LAXOR: A BNN Accelerator with Latch-XOR Logic for Local Computing

## 🟨 Contents
xxx_xxx

## 🟨 Introduction
### ◼️ Coflex optimizer framework
Coflex is a hardware-aware neural architecture search (HW-NAS) optimizer that jointly considers key parameters from the software-side neural network architecture and corresponding hardware design configurations. It operates through an iterative co-optimization framework consisting of a multi-objective Bayesian optimizer (front-end) and a performance evaluator (back-end).

In each optimization iteration, Coflex takes candidate configurations as input, evaluates their actual performance trade-offs between software accuracy (e.g., error rate) and hardware efficiency (e.g., energy-delay product), and updates the surrogate models in the Bayesian optimizer accordingly. This process enables Coflex to progressively refine the Pareto front toward a designated reference point (e.g., (0,0)) in the objective space, effectively navigating the inherent conflict between software and hardware objectives.

After multiple iterations, Coflex converges to a near-globally optimal Pareto front, where each point represents a non-dominated configuration offering an optimal trade-off between software performance and hardware cost. The final output provides interpretable architectural design recommendations for both neural network developers and hardware architects, along with the expected performance metrics of each configuration. As a result, Coflex delivers an automated, end-to-end software-hardware co-design pipeline.

<p align="center"><img width=100% src="https://github.com/myh1998/Coflex/blob/main/Figs/Fig_hw_framework_overview.png"></p>

### ◼️ Search Space Define
The search space of HW-NAS encompasses a high-dimensional hyperparameter space composed of both software-wise parameters (e.g., neural network architectural choices) and hardware-wise parameters (e.g., hardware resource configurations). To initialize the optimization process, Coflex performs uniform sampling across all dimensions of this joint search space. These sampled configurations are then used to construct the initial Gaussian surrogate models within the multi-objective Bayesian optimization front-end.

### ◼️ Total Hyper-parameters for different NAS-Benchmark suites
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
#### xxx_xxx

### ◼️ Sparse Gaussian inducing strategies
#### xxx_xxx

## 🟨 Repository File Structure

## 🟨 Installation Requirements

## 🟨 How to Run
