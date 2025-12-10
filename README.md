# Distributed Neural Network with Embeddings using MPI

## Overview

This project implements a distributed neural network training system using MPI (Message Passing Interface) for parallel processing. The model supports embedding layers for categorical features and performs distributed hyperparameter tuning through grid search.

## Features

- **Distributed Training**: Uses MPI for parallel SGD across multiple processes
- **Embedding Support**: Handles categorical features through embedding layers
- **Multiple Activation Functions**: Supports sigmoid, tanh, and ReLU activations
- **Grid Search**: Distributed hyperparameter tuning for optimal model configuration
- **Early Stopping**: Automatically stops training when loss stops improving (specifically when loss fails to decrease for 3 consecutive iterations)
- **MPI-based Parallelism**: Efficient data distribution and gradient aggregation

## Requirements

- Intel MPI 2021
- Python 3
- numpy
- matplotlib
- joblib
- mpi4py

## Installation

To set up the environment on Google Cloud VM:

1. Create a VM instance on Google Cloud Platform
2. SSH into the VM instance
3. Install Intel MPI 2021:

```bash
sudo google_install_intelmpi --impi_2021
```

4. Install Python3:

```bash
sudo apt install python3
```

5. Install required Python packages:

```bash
pip3 install --user numpy matplotlib joblib mpi4py
```

## Usage

### Single VM Execution (Multiple Processes)

```bash
mpiexec -n 4 python3 sgd_mpi.py
```

This command executes distributed training with 4 MPI processes on a single VM. You can adjust the number of processes as needed. It is recommended to use powers of 2 to facilitate local batch sampling (local batch size = global batch size / number of processes, where global batch sizes are 16, 32, 64, 128, 256).

Note: Due to memory limitations of a single VM, it may not be possible to run the full dataset. You can sample the dataset, for example by taking only the first 5000 rows, to run the code with a smaller sample.

### Multi-VM Execution (Multiple Processes Across VMs)

```bash
mpiexec -hosts <host1>,<host2>,<host3> -n 4 python3 sgd_mpi.py
```

In this configuration, host1 is responsible for loading and distributing data, so it must contain the preprocessed dataset and code files. Other hosts only need the code files.

### Data Preparation

The script we used to preprocess the data is `preprocess.py`. This script cleans and processes the NYC Taxi dataset to prepare it for later neural network training. In short, it handles large files in chunks, extracts features we want to investigate, encodes categorical features for embeddings, normalizes numeric data, and splits the dataset into training and testing sets. To use it, you need to:
1. Ensure the `nytaxi2022.csv` and the `preprocess.py` are in the same directory and go into this directory.
2. Run the command on the master process to run the script: 
    ```bash
    python3 preprocess.py
    ```
3. Running the script generates several files under the same directory, which are:
    `nytaxi_processed.npz` – preprocessed numeric and categorical data (train/test sets) including: X_train_num, X_test_num: normalized numeric features. pu_train, do_train, rate_train, pay_train (and corresponding _test): integer-indexed categorical IDs for embedding.
    `nytaxi_processed_scaler.pkl` – Trained StandardScaler object used to normalize numeric features.
    `nytaxi_processed_vocabs.pkl` – Dictionary of categorical vocabularies, mapping each unique raw ID to its integer index for: PULocationID, DOLocationID, RatecodeID, payment_type
    `nytaxi_processed_meta.pkl` – Metadata containing dataset statistics, including total samples, train/test counts, and the number of unique categories (cardinalities)
    `nytaxi_processed_numeric_features.txt` – List of numeric feature names used for model input.

In later training stage, we mainly use the `nytaxi_processed.npz` and `nytaxi_processed_meta.pkl` to get data and mete information to build and train the neural network model.

### Code Structure

#### Core Functions

- **`init_parameters()`**:  Initializes neural network weights and embedding matrices with random weight initialization and zero-initialized bias vectors
- **`forward()`**: Performs forward propagation with optional embedding lookup
- **`backward()`**: Computes gradients through backpropagation
- **`sgd()`**: Implements distributed stochastic gradient descent training
- **`grid_search_parameter()`**: Conducts distributed hyperparameter tuning
- **`calculate_rmse()`**: Computes RMSE across distributed data

#### Configurable Parameters

- **Activation Functions**: `sigmoid`, `tanh`, `relu`
- **Batch Sizes**: 16, 32, 64, 128, 256
- **Hidden Layer Sizes**: 16, 32, 64
- **Learning Rate**: Fixed at 0.01

## Implementation Details

### Data Distribution

1. The master process (rank 0) loads the complete dataset
2. Data is evenly partitioned into chunks and distributed to all MPI processes using the scatter function
3. Each process operates on its assigned data segment

### Distributed Training Process

1. Each process samples a local batch from its data partition (local batch size equals total batch size divided by number of processes)
2. Each process independently computes local loss and local gradients
3. Loss and gradients are aggregated across all processes using MPI collective operations
4. Model parameters are updated synchronously across all processes

### Grid Search Mechanism

1. The master process generates all parameter combinations
2. Each combination is evaluated through distributed training
3. Results are consolidated and the optimal configuration is selected based on test RMSE

## Output

The program generates the following outputs:
- Data distribution statistics across processes
- Grid search progress with each parameter combination
- Training and test RMSE values for each configuration
- Detailed metrics for the final optimal configuration
- Training history visualization for the best model (saved as `plot.png` in the current directory)

## Performance Considerations

- The implementation automatically balances data distribution across available processes
- Early stopping mechanism prevents overfitting and optimizes computational resources
- MPI collective operations ensure efficient gradient and loss aggregation

## Troubleshooting

- Ensure the VM with rank=0 master process contains the data files, and other VMs have the code file sgd_mpi.py
- Ensure the number of processes is compatible with the dataset size

---

# 基于MPI的分布式神经网络与嵌入层实现

## 概述

本项目实现了一个使用MPI（消息传递接口）进行并行处理的分布式神经网络训练系统。该模型支持用于分类特征的嵌入层，并通过网格搜索执行分布式超参数调优。

## 功能特点

- **分布式训练**：使用MPI在多个进程间进行并行SGD
- **嵌入层支持**：通过嵌入层处理分类特征
- **多种激活函数**：支持sigmoid、tanh和ReLU激活函数
- **网格搜索**：分布式超参数调优以获得最优模型配置
- **早停机制**：当损失不再改善时自动停止训练（具体表现为损失连续3次不再下降）
- **基于MPI的并行**：高效的数据分布和梯度聚合

## 环境要求

- Intel MPI 2021
- Python 3
- numpy
- matplotlib
- joblib
- mpi4py

## 安装步骤

在Google Cloud虚拟机中配置环境：

1. 在Google云平台上创建虚拟机实例
2. 通过SSH连接到虚拟机实例
3. 安装Intel MPI 2021：

```bash
sudo google_install_intelmpi --impi_2021
```

4. 安装Python3：

```bash
sudo apt install python3
```

5. 安装所需的Python包：

```bash
pip3 install --user numpy matplotlib joblib mpi4py
```

## 使用方法

### 基本执行（在单个VM中使用多进程运行）

```bash
mpiexec -n 4 python3 sgd_mpi.py
```

此命令在一个VM上使用4个MPI进程执行分布式训练。您可以根据需要调整进程数量，建议选用2的幂，以便于处理各进程的本地采样批次（本地采样批次=全局采样批次/进程数，全局采样批次为16,32,64,128,256）

注：由于单VM内存限制，通常无法运行完整数据集，可以对数据集进行采样，例如只取数据集前5000行，使用小样本运行代码

### 多VM多进程执行

```bash
mpiexec -hosts <host1>,<host2>,<host3> -n 4 python3 sgd_mpi.py
```

其中host1的rank 0进程负责加载和分配数据，因此需要包含预处理后的数据集和代码文件，其他host只需要代码文件

### 数据准备

代码需要两个数据文件，需要先下载源数据集并通过运行`preprocess.py`生成以下数据文件（该Python脚本用于数据预处理，包括嵌入生成和将数据集按7:3比例拆分为训练集和测试集）：

- `./embedding/nytaxi_processed.npz`：包含处理后的数值和分类特征
- `./embedding/nytaxi_processed_meta.pkl`：包含分类特征基数的元数据

### 代码结构

#### 核心函数

- **`init_parameters()`**：初始化神经网络权重和嵌入矩阵，权重随机初始化，偏置初始化为零向量
- **`forward()`**：执行前向传播，包含可选的嵌入查找
- **`backward()`**：通过反向传播计算梯度
- **`sgd()`**：实现分布式随机梯度下降训练
- **`grid_search_parameter()`**：执行分布式超参数调优
- **`calculate_rmse()`**：计算分布式数据的RMSE

#### 可配置参数

- **激活函数**：`sigmoid`、`tanh`、`relu`
- **批处理大小**：16、32、64、128、256
- **隐藏层大小**：16、32、64
- **学习率**：固定为0.01

## 实现细节

### 数据分布

1. 主进程（rank 0）加载完整数据集
2. 使用scatter函数，数据被均匀分割成块并分布到所有MPI进程
3. 每个进程处理其分配的数据段

### 分布式训练流程

1. 每个进程从其数据分区中采样本地批次（本地批次等于总批次除以进程数目）
2. 各进程独立计算本地损失以及本地梯度
3. 使用MPI集合操作在所有进程间聚合损失和梯度
4. 所有进程同步更新模型参数

### 网格搜索机制

1. 主进程生成所有参数组合
2. 通过分布式训练评估每个组合
3. 整合结果并根据测试RMSE选择最优配置

## 输出结果

程序生成以下输出：

- 跨进程的数据分布统计
- 每个参数组合的网格搜索进度
- 每种配置的训练和测试RMSE值
- 最终最优配置的详细指标
- 最佳模型的训练历史可视化（保存在当前目录下`plot.png`）

## 故障排除

- 确保rank 0主进程所在的VM拥有数据文件，其他VM拥有代码文件sgd_mpi.py
- 确保进程数量与数据集大小兼容
