# ======================= CONFIG PARAMETERS =======================
data_ = '../data/wpbc.data'
scalers = [None, 'MinMax', 'Standard']
params = {
        'activation': ('identity', 'logistic', 'tanh', 'relu'),
        'solver': ('lbfgs', 'sgd', 'adam'),
        'alpha': [10**(-x) for x in range(0, -4, -1)],
        'learning_rate': ('constant', 'invscaling', 'adaptive')
    }


# ==================== EXPERIMENTS ============================
experiment_one = {
    'response': 'outcome',
    'model_type': 'classifier',
    'scaler': None,
    'num_iterations': 10
}