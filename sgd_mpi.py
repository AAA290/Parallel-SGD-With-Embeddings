# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import joblib
import time
from mpi4py import MPI
import argparse

# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)

def tanh(x):
    return np.tanh(x)

def d_tanh(tanh_output):
    return 1 - tanh_output ** 2

def relu(x):
    return np.maximum(0, x)

def d_relu(relu_output):
    return (relu_output > 0).astype(np.float32)

# Initialize parameter matrix theta (including embedding parameters)
def init_parameters(hidden_size, input_size, embedding_config=None):
    """
    embedding_config: dictionary containing embedding configuration
        {'pu': (vocab_size, emb_dim), 'do': (vocab_size, emb_dim), ...}
    """
    theta = {}
    
    np.random.seed(47)
    
    # If there is embedding configuration, initialize embedding matrices
    if embedding_config:
        theta['embeddings'] = {}
        for name, (vocab_size, emb_dim) in embedding_config.items():
            theta['embeddings'][name] = np.random.randn(vocab_size, emb_dim) * 0.01
        # Adjust input dimension: numerical feature dimension + sum of all embedding dimensions
        total_emb_dim = sum(emb_dim for _, (_, emb_dim) in embedding_config.items())
        actual_input_size = input_size + total_emb_dim
    else:
        actual_input_size = input_size
    
    # Hidden layer weights (hidden_size, actual_input_size)
    theta['w'] = np.random.randn(hidden_size, actual_input_size) * 0.01
    # Hidden layer bias (hidden_size, 1)
    theta['b'] = np.zeros((hidden_size, 1))
    
    # Output layer weights (1, hidden_size)
    theta['v'] = np.random.randn(1, hidden_size) * 0.01
    # Output layer bias (1, 1)
    theta['b2'] = np.zeros((1, 1))
    
    return theta

# Forward propagation (x_numeric is numerical features)
def forward(theta, x_numeric, categorical_data=None, activation_function='sigmoid'):
    # If there is categorical data, perform embedding lookup and concatenation
    if categorical_data and 'embeddings' in theta:
        emb_features = []
        for name, idx in categorical_data.items():
            if name in theta['embeddings']:
                emb_vec = theta['embeddings'][name][idx]  # Get embedding vector
                emb_features.append(emb_vec)
        
        if emb_features:
            # Concatenate all embedding features
            emb_combined = np.concatenate(emb_features)
            # Concatenate with numerical features
            x_combined = np.concatenate([x_numeric, emb_combined])
        else:
            x_combined = x_numeric
    else:
        x_combined = x_numeric
    
    x_combined = x_combined.reshape(-1, 1) # Convert to column vector
    
    Z = np.dot(theta['w'], x_combined) + theta['b']
    
    if activation_function == 'sigmoid': 
        A = sigmoid(Z)
    elif activation_function == 'tanh': 
        A = tanh(Z)
    else:
        A = relu(Z)
    
    y_pred = np.dot(theta['v'], A) + theta['b2']
    return y_pred[0, 0], A, Z, x_combined

# Backward propagation
def backward(theta, x_numeric, categorical_data, A, Z, x_combined, activation_function='sigmoid'):
    gradients = {}
    
    # Calculate derivative based on activation function
    if activation_function == 'sigmoid':
        d_activation = d_sigmoid(A)
    elif activation_function == 'tanh':
        d_activation = d_tanh(A)
    else:
        d_activation = d_relu(A)
    
    # Original gradient calculation
    x_combined_T = x_combined.reshape(1, -1) # Convert to row vector
    
    temp = d_activation * theta['v'].T  # (hidden_size, 1)
    gradients['w'] = np.dot(temp, x_combined_T) 
    gradients['b'] = temp
    
    gradients['v'] = A.T
    gradients['b2'] = np.array([[1]]) 
    
    # Calculate gradient with respect to input, for embedding update
    dX_combined = np.dot(theta['w'].T, temp)  # (input_size, 1)
    dX_combined = dX_combined.flatten()
    
    # If there is categorical data, calculate embedding gradients
    gradients['embeddings'] = {}
    if categorical_data and 'embeddings' in theta:
        num_numeric = len(x_numeric)
        current_idx = num_numeric
        
        for name, idx in categorical_data.items():
            if name in theta['embeddings']:
                emb_dim = theta['embeddings'][name].shape[1]
                # Get the gradient part corresponding to this embedding
                d_emb = dX_combined[current_idx:current_idx + emb_dim]
                # Initialize gradient matrix for this embedding
                grad_matrix = np.zeros_like(theta['embeddings'][name])
                # Set gradient at corresponding position
                grad_matrix[idx] = d_emb
                gradients['embeddings'][name] = grad_matrix
                current_idx += emb_dim
    
    return gradients

# Update parameter dictionary
def update_dict(dic_0, op, num, dic_1):
    # Process regular parameters
    if op == '+':
        dic_0['w'] = dic_0['w'] + num * dic_1['w']
        dic_0['b'] = dic_0['b'] + num * dic_1['b']        
        dic_0['v'] = dic_0['v'] + num * dic_1['v']        
        dic_0['b2'] = dic_0['b2'] + num * dic_1['b2']
        
        # Update embedding parameters
        if 'embeddings' in dic_1 and 'embeddings' in dic_0:
            for emb_name in dic_1['embeddings']:
                if emb_name not in dic_0['embeddings']:
                    dic_0['embeddings'][emb_name] = np.zeros_like(dic_1['embeddings'][emb_name])
                dic_0['embeddings'][emb_name] = dic_0['embeddings'][emb_name] + num * dic_1['embeddings'][emb_name]
                    
    elif op == '-':
        dic_0['w'] = dic_0['w'] - num * dic_1['w']
        dic_0['b'] = dic_0['b'] - num * dic_1['b']        
        dic_0['v'] = dic_0['v'] - num * dic_1['v']        
        dic_0['b2'] = dic_0['b2'] - num * dic_1['b2']
        
        # Update embedding parameters
        if 'embeddings' in dic_1 and 'embeddings' in dic_0:
            for emb_name in dic_1['embeddings']:
                if emb_name not in dic_0['embeddings']:
                    dic_0['embeddings'][emb_name] = np.zeros_like(dic_1['embeddings'][emb_name])
                dic_0['embeddings'][emb_name] = dic_0['embeddings'][emb_name] - num * dic_1['embeddings'][emb_name]
                    
    elif op == '/':
        # Division for calculating average
        dic_0['w'] = dic_1['w'] / num
        dic_0['b'] = dic_1['b'] / num        
        dic_0['v'] = dic_1['v'] / num       
        dic_0['b2'] = dic_1['b2'] / num
        
        # Process embedding parameters
        if 'embeddings' in dic_1:
            dic_0['embeddings'] = {}
            for emb_name in dic_1['embeddings']:
                dic_0['embeddings'][emb_name] = dic_1['embeddings'][emb_name] / num
    else:
        print('Invalid op input, returning dic_0')
    return dic_0

# Initialize gradient dictionary
def init_gradient_dict(theta):
    grad_dict = {
        'w': np.zeros_like(theta['w']),
        'b': np.zeros_like(theta['b']),
        'v': np.zeros_like(theta['v']),
        'b2': np.zeros_like(theta['b2'])
    }
    
    if 'embeddings' in theta:
        grad_dict['embeddings'] = {}
        for emb_name, emb_matrix in theta['embeddings'].items():
            grad_dict['embeddings'][emb_name] = np.zeros_like(emb_matrix)
    
    return grad_dict

# SGD training
def sgd(batch_size, x_train, y_train, theta_0, learning_rate, 
        categorical_data=None, max_iterations=1000, activation_function='sigmoid', comm=None):
    """
    categorical_data: list, each element is a dictionary containing categorical data indices for a sample
    """
    rank = comm.Get_rank()
    n = comm.Get_size()
    
    theta = theta_0
    iteration = 0
    dataset_size = len(y_train)
    average_loss = 10000 # Arbitrary large number
    history = []
    patience = 0
    
    for _ in range(max_iterations):
        iteration += 1
        
        # Randomly sample batch_size//n (n is number of processes, each process samples average, total samples is Batch_size)
        local_batch_size  = batch_size//n
        # If local_batch_size is 0, set it to 1 to avoid sampling errors on small datasets
        if local_batch_size == 0:
            local_batch_size = 1
        
        if dataset_size < local_batch_size: # Handle case where local data is smaller than local batch size
            sample_indices = np.random.choice(range(dataset_size), size=dataset_size, replace=False)
            local_batch_size = dataset_size # Adjust local_batch_size for loss calculation
        else:
            sample_indices = np.random.choice(range(dataset_size), size=local_batch_size, replace=False)

        batch_x = x_train[sample_indices]
        batch_y = y_train[sample_indices]
        
        # Get corresponding categorical data (if any)
        batch_categorical = None
        if categorical_data is not None:
            batch_categorical = [categorical_data[i] for i in sample_indices]
        
        batch_loss = 0
        batch_gradient = init_gradient_dict(theta)  # Use initialization function
        
        for i in range(local_batch_size):
            # Get categorical data (if any)
            cat_data = batch_categorical[i] if batch_categorical else None
            
            # Forward propagation (prediction)
            prediction, A, Z, x_combined = forward(
                theta, batch_x[i], cat_data, activation_function
            )
            
            # Calculate error
            error = prediction - batch_y[i]
            # Accumulate loss
            batch_loss += error ** 2
            
            # Backward propagation (derivative)
            gradients = backward(theta, batch_x[i], cat_data, A, Z, x_combined, activation_function)
            
            # Accumulate gradients
            temp_grad = {}
            for key in ['w', 'b', 'v', 'b2']:
                temp_grad[key] = error * gradients[key]
            
            # Process embedding gradients
            temp_grad['embeddings'] = {}
            if 'embeddings' in gradients:
                for emb_name in gradients['embeddings']:
                    temp_grad['embeddings'][emb_name] = error * gradients['embeddings'][emb_name]
            
            # Accumulate gradients
            batch_gradient = update_dict(batch_gradient, '+', 1, temp_grad)
        
        # Global aggregation of loss and gradients
        global_batch_loss = comm.allreduce(batch_loss, op=MPI.SUM)
        # Sum up actual sampled local batch sizes
        global_batch_size = comm.allreduce(np.array([local_batch_size]), op=MPI.SUM)[0] 
        current_average_loss = global_batch_loss / (2 * global_batch_size)
        
        # Aggregate gradients
        global_gradient = init_gradient_dict(theta)
        # Aggregate regular parameter gradients
        for key in ['w', 'b', 'v', 'b2']:
            local_grad = batch_gradient[key]
            global_grad = np.zeros_like(local_grad)
            comm.Allreduce(local_grad, global_grad, op=MPI.SUM)
            global_gradient[key] = global_grad / global_batch_size  # Average gradient
        # Aggregate embedding gradients
        if 'embeddings' in batch_gradient:
            global_gradient['embeddings'] = {}
            for emb_name in batch_gradient['embeddings']:
                local_emb_grad = batch_gradient['embeddings'][emb_name]
                global_emb_grad = np.zeros_like(local_emb_grad)
                comm.Allreduce(local_emb_grad, global_emb_grad, op=MPI.SUM)
                global_gradient['embeddings'][emb_name] = global_emb_grad / global_batch_size
                
        # Early stopping mechanism (stop if loss doesn't decrease for 15 consecutive times)
        stop_flag = np.array([0])
        if rank == 0:
            if current_average_loss >= average_loss:
                patience += 1
                if patience >= 15:
                    stop_flag[0] = 1
            else:
                patience = 0
                average_loss = current_average_loss
        
        # Broadcast stop flag
        stop_flag = comm.bcast(stop_flag, root=0)
        if stop_flag[0]:
            if rank == 0: print(f"Early stopping at iteration {iteration}, final loss: {average_loss:.5f}")
            break
        
        # Parameter update (all processes update)
        theta = update_dict(theta, '-', learning_rate, global_gradient)
        
        if rank == 0: history.append((iteration, average_loss))
    
    if rank == 0 and iteration == max_iterations:
        print(f"Reached maximum iterations {max_iterations}, final loss: {average_loss:.5f}")
    
    return theta, history

# Calculate RMSE
def calculate_rmse(theta, x, y, categorical_data, activation_function, comm):
    rank = comm.Get_rank()
    
    # Local predictions and errors
    local_predictions = []
    local_errors = []
    
    for i, x_i in enumerate(x):
        cat_data = categorical_data[i] if categorical_data else None
        pred, _, _, _ = forward(theta, x_i, cat_data, activation_function)
        local_predictions.append(pred)
        local_errors.append((pred - y[i]) ** 2)
    
    local_total_error = np.sum(local_errors)
    local_sample_count = len(y)
    
    # Option 2: reduce
    if rank == 0: # Only calculate in rank0 process
        # Prepare receive buffers
        global_total_error = np.array([0.0])
        global_sample_count = np.array([0])
        
        # Receive error sum from all processes
        comm.Reduce(np.array([local_total_error]), global_total_error, op=MPI.SUM, root=0)
        # Receive sample count from all processes
        comm.Reduce(np.array([local_sample_count]), global_sample_count, op=MPI.SUM, root=0)
    
        # Calculate global RMSE
        mse = global_total_error[0] / global_sample_count[0]
        rmse = np.sqrt(mse)
    else: # Other processes: only send data
        comm.Reduce(np.array([local_total_error]), None, op=MPI.SUM, root=0)
        comm.Reduce(np.array([local_sample_count]), None, op=MPI.SUM, root=0)
        rmse = 0
        
    # there is no need for other processes to get the value of rmse
    # rmse = comm.bcast(rmse, root=0)

    return rmse

def grid_search_parameter(X_train, y_train, X_test, y_test, meta, train_categorical_data, test_categorical_data, comm, combinations):
    rank = comm.Get_rank()
    n = comm.Get_size()
    
    # Extract training data dimensions and set fixed parameters
    num_dim = X_train.shape[1]
    learning_rate = 0.01
    embedding_config = {
        'pu': (meta['pu_cardinality'], 16),
        'do': (meta['do_cardinality'], 16), 
        'rate': (meta['rate_cardinality'], 4),
        'pay': (meta['pay_cardinality'], 4)
    }
    
    # Master process manages parameter combinations and results
    if rank == 0:
        all_combinations = combinations
        results = []
        num_combinations = len(all_combinations)
        print(f"Starting distributed parameter tuning, total {num_combinations} parameter combinations, using {n} processes")
    else:
        all_combinations = None
        results = None
        num_combinations = None
    
    # Broadcast number of parameter combinations
    num_combinations = comm.bcast(num_combinations, root=0)
    
    # Test parameter combinations one by one (all processes participate in the loop)
    for combo_idx in range(num_combinations):
        # Master process selects current parameter combination and broadcasts
        if rank == 0:
            # current_combo is a tuple: (act_func, batch_size, hidden_size)
            current_combo = all_combinations[combo_idx]
            act_func, batch_size, hidden_size = current_combo
            print(f"Testing combination {combo_idx+1}/{num_combinations}: "
                  f"Activation function={act_func}, Batch size={batch_size}, Hidden size={hidden_size}")
        else:
            current_combo = None
        
        # Broadcast current parameter combination to all processes
        current_combo = comm.bcast(current_combo, root=0)
        act_func, batch_size, hidden_size = current_combo
        
        # Initialize parameters (master process initializes and broadcasts to all processes)
        if rank == 0:
            theta_0 = init_parameters(hidden_size, num_dim, embedding_config)
        else:
            theta_0 = None
        
        # Broadcast initial parameters
        # Use joblib.dump/load to serialize complex numpy dict for mpi4py broadcast/scatter
        # For simplicity and small size, standard Python/MPI Bcast is used here, assuming dict is serializable
        theta_0 = comm.bcast(theta_0, root=0)
        
        # Distributed training (all processes participate)
        start_time = time.time()
        optimized_theta, history = sgd(
            batch_size, X_train, y_train, theta_0, learning_rate,
            categorical_data=train_categorical_data, max_iterations=500, # Use fixed max_iterations
            activation_function=act_func, comm=comm
        )
        training_time = time.time() - start_time
        
        # Distributed RMSE calculation (all processes participate)
        train_rmse = calculate_rmse(
            optimized_theta, X_train, y_train, 
            train_categorical_data, act_func, comm
        )
        
        test_rmse = calculate_rmse(
            optimized_theta, X_test, y_test,
            test_categorical_data, act_func, comm
        )
        
        # Only master process saves results
        if rank == 0:
            result = {
                'activation_function': act_func,
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'training_time': training_time,
                'final_loss': history[-1][1] if history else float('inf'),
                'convergence_iter': len(history),
                'history': history
            }
            results.append(result)
            
            print(f"Result: Train_RMSE={train_rmse:.4f}, Test_RMSE={test_rmse:.4f}, "
                  f"Time={training_time:.2f}s\n")
    
    return results if rank == 0 else None

# Analyze results, find optimal combination
def analyze(results):
    # Find combination with minimum test RMSE
    best_result = min(results, key=lambda x: x['test_rmse'])
    
    print("\n" + "="*50)
    print("Best configuration result:")
    print("="*50)
    print(f"Activation function: {best_result['activation_function']}")
    print(f"Batch size: {best_result['batch_size']}")
    print(f"Hidden layer size: {best_result['hidden_size']}")
    print(f"Train RMSE: {best_result['train_rmse']:.4f}")
    print(f"Test RMSE: {best_result['test_rmse']:.4f}")
    print(f"Training time: {best_result['training_time']:.2f} seconds")
    print(f"Convergence iterations: {best_result['convergence_iter']}")
    
    # Plot training history for best combination
    k, Rk = zip(*best_result['history'])
    plt.plot(k, Rk, 'o-', markersize=2)
    plt.xlabel('k')
    plt.ylabel('R(theta_k)')
    plt.title(f"Training History ({best_result['activation_function']}, BS={best_result['batch_size']}, HS={best_result['hidden_size']})")
    # plt.show()
    plt.savefig('./plot_best.png', dpi=150)
    plt.clf() 

# New function for single run analysis
def analyze_single_run(result):
    print("\n" + "="*50)
    print("Single run result:")
    print("="*50)
    print(f"Activation function: {result['activation_function']}")
    print(f"Batch size: {result['batch_size']}")
    print(f"Hidden layer size: {result['hidden_size']}")
    print(f"Train RMSE: {result['train_rmse']:.4f}")
    print(f"Test RMSE: {result['test_rmse']:.4f}")
    print(f"Training time: {result['training_time']:.2f} seconds")
    print(f"Convergence iterations: {result['convergence_iter']}")
    print(f"Final Loss: {result['final_loss']:.5f}")
    
    # Plot training history
    if result['history']:
        k, Rk = zip(*result['history'])
        plt.plot(k, Rk, 'o-', markersize=2)
        plt.xlabel('k')
        plt.ylabel('R(theta_k)')
        plt.title(f"Training History ({result['activation_function']}, BS={result['batch_size']}, HS={result['hidden_size']})")
        plt.savefig('./plot_single_run.png', dpi=150)
        plt.clf()

def load_data(data_file_path, meta_file_path, sample_size=None):
    # Load data
    data = np.load(data_file_path)
    meta = joblib.load(meta_file_path)
    
    # Categorical feature names
    categorical_features = ['pu', 'do', 'rate', 'pay']
    
    # Extract training and test sets
    datasets = {}
    for split in ['train', 'test']:
        # Numerical features
        datasets[f'X_{split}_num'] = data[f"X_{split}_num"]
        datasets[f'y_{split}'] = data[f"y_{split}"]
        
        # Categorical features
        for feature in categorical_features:
            datasets[f'{feature}_{split}'] = data[f"{feature}_{split}"]
    
    # If only taking a portion
    if sample_size:
        train_ratio = 0.7 # train:test = 7:3
        actual_data_size = len(datasets['y_train']) + len(datasets['y_test'])
        sample_size = min(sample_size, actual_data_size) 
        
        new_train_size = int(round(sample_size * train_ratio))
        new_test_size = sample_size - new_train_size

        for key in datasets:
            if key.endswith('_train'):
                datasets[key] = datasets[key][:new_train_size]
            elif key.endswith('_test'):
                datasets[key] = datasets[key][:new_test_size]
    
    return datasets, meta, categorical_features

def create_categorical_dicts(datasets, categorical_features):
    """Create categorical data dictionaries for training and test sets"""
    categorical_dicts = {}
    
    for split in ['train', 'test']:
        split_size = len(datasets[f'y_{split}']) 
        
        categorical_dicts[split] = []
        
        for i in range(split_size):
            cat_dict = {}
            for feature in categorical_features:
                cat_dict[feature] = datasets[f'{feature}_{split}'][i]
            categorical_dicts[split].append(cat_dict)
    
    return categorical_dicts['train'], categorical_dicts['test']

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Distributed SGD training with optional Grid Search.")
    
    # Options for parameter setting / single run
    parser.add_argument('--activation_function', type=str, default='sigmoid', 
                        choices=['sigmoid', 'tanh', 'relu'],
                        help="Activation function to use for the single run.")
    parser.add_argument('--hidden_size', type=int, default=32, 
                        help="Hidden layer size for the single run.")
    parser.add_argument('--batch_size', type=int, default=64, 
                        help="Global batch size for the single run.")
    
    # Control variable for grid search
    parser.add_argument('--grid_search', type=str, default='True',
                        help="Set to 'None' to skip grid search and use the single set of parameters. Otherwise, performs grid search.")
    
    # Data sampling control
    parser.add_argument('--sample', type=str, default='None',
                        help="Number of total samples to use (train + test). Set to 'None' to use all data.")
    
    return parser.parse_args()


if __name__ == "__main__":
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n = comm.Get_size() # Number of processes
    
    # Parse command line arguments
    args = parse_args()

    # Only master process loads complete data
    if rank == 0:
        if args.sample == 'None':
            sample_size = None
        else:
            try:
                sample_size = int(args.sample)
            except ValueError:
                sample_size = 2000
                print('sample_size ValueError, use default 2000')
        
        datasets, meta, categorical_features = load_data("./embedding/nytaxi_processed.npz", "./embedding/nytaxi_processed_meta.pkl", sample_size)
        
        # Create categorical data dictionaries
        train_categorical_data, test_categorical_data = create_categorical_dicts(
            datasets, categorical_features
        )
        
        # Complete data
        X_train_full = datasets['X_train_num']
        y_train_full = datasets['y_train']
        X_test_full = datasets['X_test_num']
        y_test_full = datasets['y_test']
        
        # Calculate data distribution
        total_train_size = len(y_train_full)
        train_chunk_size = total_train_size // n
        remainder = total_train_size % n
        
        # Prepare data chunks
        train_data_chunks = []
        test_data_chunks = []
        
        start_idx = 0
        for i in range(n):
            current_train_size = train_chunk_size + (1 if i < remainder else 0)
            
            # Test data distribution: Ensure test data is also split fairly among processes
            total_test_size = len(y_test_full)
            test_chunk_size = total_test_size // n
            test_remainder = total_test_size % n
            
            test_start = i * test_chunk_size + min(i, test_remainder)
            current_test_size = test_chunk_size + (1 if i < test_remainder else 0)
            test_end = test_start + current_test_size
            
            # Training data chunk (same as original: split non-contiguously)
            train_chunk = {
                'X': X_train_full[start_idx:start_idx + current_train_size],
                'y': y_train_full[start_idx:start_idx + current_train_size],
                'categorical': train_categorical_data[start_idx:start_idx + current_train_size]
            }
            train_data_chunks.append(train_chunk)
            
            # Test data chunk (corrected distribution calculation)
            test_chunk = {
                'X': X_test_full[test_start:test_end],
                'y': y_test_full[test_start:test_end],
                'categorical': test_categorical_data[test_start:test_end]
            }
            test_data_chunks.append(test_chunk)
            
            start_idx += current_train_size
    else:
        train_data_chunks = None
        test_data_chunks = None
        meta = None
    
    # Broadcast metadata
    meta = comm.bcast(meta, root=0)
    
    # Scatter data to each process
    local_train_data = comm.scatter(train_data_chunks, root=0)
    local_test_data = comm.scatter(test_data_chunks, root=0)
    
    # Extract local data
    X_train_local = local_train_data['X']
    y_train_local = local_train_data['y']
    train_categorical_local = local_train_data['categorical']
    
    X_test_local = local_test_data['X']
    y_test_local = local_test_data['y']
    test_categorical_local = local_test_data['categorical']
    
    # All processes participate in gather operation
    local_train_size = np.array([len(y_train_local)])
    local_test_size = np.array([len(y_test_local)])
    
    if rank == 0:
        all_train_sizes = np.zeros(n, dtype=int)
        all_test_sizes = np.zeros(n, dtype=int)
    else:
        all_train_sizes = None
        all_test_sizes = None
    
    comm.Gather(local_train_size, all_train_sizes, root=0)
    comm.Gather(local_test_size, all_test_sizes, root=0)
    
    if rank == 0:
        print(f"Data distribution verification:")
        print(f"  Training data per process: {all_train_sizes.tolist()}")
        print(f"  Test data per process: {all_test_sizes.tolist()}")
        print(f"  Total training samples: {sum(all_train_sizes)}")
        print(f"  Total test samples: {sum(all_test_sizes)}")
    
    # Rank 0 delete complete data references
    if rank == 0:
        del X_train_full, y_train_full, train_categorical_data
        del X_test_full, y_test_full, test_categorical_data
        print(f"Rank 0: Data distributed, local data size: {len(y_train_local)}")
    
    print(f"Process {rank}: Training data size = {len(y_train_local)}, Test data size = {len(y_test_local)}")
    
    # Synchronize
    comm.Barrier()
  
    if rank == 0:
        if args.grid_search == 'None':
            # Single run mode: create a list with one combination from arguments
            combinations = [
                (args.activation_function, args.batch_size, args.hidden_size)
            ]
            print(f"Running single set of parameters: Activation={args.activation_function}, Batch Size={args.batch_size}, Hidden Size={args.hidden_size}")
        else:
            # Grid search mode: generate all combinations
            print(f"Running full Grid Search.")
            activation_functions = ['sigmoid', 'tanh', 'relu']
            batch_sizes = [16, 32, 64, 128, 256]
            hidden_sizes = [16, 32, 64]
            combinations = []
            for act_func in activation_functions:
                for batch_size in batch_sizes:
                    for hidden_size in hidden_sizes:
                        combinations.append((act_func, batch_size, hidden_size))
    else:
        combinations = None
        
    # Broadcast combinations to all processes
    combinations = comm.bcast(combinations, root=0)
    
    # Distributed parameter tuning
    results = grid_search_parameter(
        X_train_local, y_train_local, X_test_local, y_test_local,
        meta, train_categorical_local, test_categorical_local, comm, combinations
    )
    
    # Only master process analyzes results
    if rank == 0:
        if args.grid_search == 'None':
            analyze_single_run(results[0]) 
        else:
            analyze(results)
    
    MPI.Finalize()