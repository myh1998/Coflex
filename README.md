# LAXOR: A BNN Accelerator with Latch-XOR Logic for Local Computing

## 🟨 Contents
xxx_xxx

## 🟨 Introduction
### ◼️ Coflex optimizer framework
Coflex is a hardware-aware neural architecture search (HW-NAS) optimizer that jointly considers key parameters from the software-side neural network architecture and corresponding hardware design configurations. It operates through an iterative co-optimization framework consisting of a multi-objective Bayesian optimizer (front-end) and a performance evaluator (back-end).

In each optimization iteration, Coflex takes candidate configurations as input, evaluates their actual performance trade-offs between software accuracy (e.g., error rate) and hardware efficiency (e.g., energy-delay product), and updates the surrogate models in the Bayesian optimizer accordingly. This process enables Coflex to progressively refine the Pareto front toward a designated reference point (e.g., (0,0)) in the objective space, effectively navigating the inherent conflict between software and hardware objectives.

After multiple iterations, Coflex converges to a near-globally optimal Pareto front, where each point represents a non-dominated configuration offering an optimal trade-off between software performance and hardware cost. The final output provides interpretable architectural design recommendations for both neural network developers and hardware architects, along with the expected performance metrics of each configuration. As a result, Coflex delivers an automated, end-to-end software-hardware co-design pipeline.

<p align="center"><img width=60% src="[https://github.com/myh1998/Coflex/blob/main/Figs/Fig_hw_framework_overview.png]"></p>

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
