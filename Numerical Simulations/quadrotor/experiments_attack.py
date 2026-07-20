import os
import sys
import torch
import random
import hashlib
import argparse
import numpy as np
from scipy import io as spio
from datetime import datetime
from attack import CyberAttack
import matplotlib.pyplot as plt
from utils import DualOutput, matlab_safe_name
import quadrotor, controller, trajectory, plot, utils

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run quadrotor experiments with attacks.')
parser.add_argument('--test_name', type=str, default='Fig8', choices=['Fig8', 'Hover', 'SpiralUp'], help='Test trajectory name')
parser.add_argument('--num_train_experiments', type=int, default=3, help='Number of training experiments')
parser.add_argument('--num_test_experiments', type=int, default=10, help='Number of testing experiments')
parser.add_argument('--seed', type=int, default=51, help='Random seed')
parser.add_argument('--test_max_wind', type=float, default=8.0, help='Maximum wind speed in test distributions')
parser.add_argument('--attack_intensity', type=float, default=0.8, help='Attack intensity')
parser.add_argument('--attack_frequency', type=float, default=0.4, help='Attack frequency')
parser.add_argument('--attack_duration', type=float, default=4.0, help='Attack duration key')
parser.add_argument('--attack_start_time', type=float, default=2.0, help='Attack start time')
parser.add_argument('--show_attack', action='store_true', help='Show attacked control and state signals')

args = parser.parse_args()

test_name = args.test_name                          # {'Fig8', 'Hover', 'SpiralUp'}
num_train_experiments = args.num_train_experiments  # number of experiments to run
num_test_experiments = args.num_test_experiments    # number of test experiments to run
seed = args.seed                                    # random seed for reproducibility
test_max_wind = args.test_max_wind                  # maximum wind speed in test distributions: 0.0, 2.0, 4.0, 6.0, 8.0, 10.0
attack_intensity = args.attack_intensity            # attack intensity: 0.2, 0.4, 0.6, 0.8
attack_frequency = args.attack_frequency            # attack frequency: 0.2, 0.4
attack_duration = args.attack_duration              # attack duration
attack_start_time = args.attack_start_time          # attack start time
show_attack = args.show_attack                      # show attacked control and state signals

attack_types = ['none', 'sensor_bias', 'actuator_bias', 'sensor_false_data_injection', 'actuator_false_data_injection', 'sensor_dos', 'actuator_dos', 'sensor_delay', 'actuator_delay'] # type of cyber attack: 'sensor_bias', 'actuator_bias', 'sensor_false_data_injection', 'actuator_false_data_injection', 'sensor_dos', 'actuator_dos', 'sensor_delay', 'actuator_delay', 'none'
attack_type = attack_types[3]  # select attack type

if test_name == 'Hover':
    T = trajectory.Hover
    t_kwargs = {
        'pd' : np.zeros(3)
    }
    time_stop = 6.0
elif test_name == 'Fig8':
    T = trajectory.Fig8
    t_kwargs = {
        'T': 10.0
    }
    time_stop = 10.0
elif test_name == 'SpiralUp':
    T = trajectory.SpiralUp
    t_kwargs = {
        'T': 10.0
    }
    time_stop = 10.0
used_trajectory = T(**t_kwargs)

eta_a = 0.01
eta_A_threshold_convex = 0.0001
eta_A_convex = eta_A_threshold_convex * 1
eta_A_biconvex = 0.0001
eta_A_deep = 0.015
dim_a = 30
dim_A = 150
layer_sizes = (40, 80)
feature_freq = 0.25

# Generate timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create main results folder
results_main = 'results_quadrotor'
os.makedirs(results_main, exist_ok=True)

# Create subfolder with experiment parameters
attack_type_for_name = attack_type
folder_name = f'{test_name}_{attack_type_for_name}_{os.path.basename(__file__)[4:-3]}_atk{attack_intensity}_f{attack_frequency}_d{attack_duration}_t{attack_start_time}_tr{num_train_experiments}_te{num_test_experiments}_s{seed}_w{test_max_wind}'
results_dir_candidate = os.path.join(results_main, folder_name)
if os.name == 'nt' and len(os.path.abspath(results_dir_candidate)) > 120:
    attack_type_for_name = f"{hashlib.md5(attack_type.encode('utf-8')).hexdigest()[:8]}"
    folder_name = f'{test_name}_{attack_type_for_name}_{os.path.basename(__file__)[4:-3]}_atk{attack_intensity}_f{attack_frequency}_d{attack_duration}_t{attack_start_time}_tr{num_train_experiments}_te{num_test_experiments}_s{seed}_w{test_max_wind}'

results_dir = os.path.join(results_main, folder_name)
os.makedirs(results_dir, exist_ok=True)

# File name with all parameters
file_base = f'{timestamp}_tr{num_train_experiments}_te{num_test_experiments}_s{seed}_w{test_max_wind}'

# Create timestamped log file
log_file = open(results_dir+f'/{file_base}_output.log', 'w', encoding='utf-8')
sys.stdout = DualOutput(log_file, sys.__stdout__)
sys.stderr = DualOutput(log_file, sys.__stderr__)

print(f"Running experiments with settings:")
print(f"  script_name: {os.path.basename(__file__)}")
print(f"  test_name: {test_name}")
print(f"  num_train_experiments: {num_train_experiments}")
print(f"  num_test_experiments: {num_test_experiments}")
print(f"  seed: {seed}")
print(f"  test_max_wind: {test_max_wind}")
print(f"  attack_intensity: {attack_intensity}")
print(f"  attack_frequency: {attack_frequency}")
print(f"  attack_duration: {attack_duration}")
print(f"  attack_start_time: {attack_start_time}")
print(f"  show_attack: {show_attack}")
print(f"  attack_type: {attack_type}")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

CYBERATTACKERS = {
    'baseline': CyberAttack(attack_type=attack_type, attack_intensity=attack_intensity, attack_frequency=attack_frequency, attack_duration=attack_duration, attack_start_time=attack_start_time),
    'baseline-pure': CyberAttack(attack_type=attack_type, attack_intensity=attack_intensity, attack_frequency=attack_frequency, attack_duration=attack_duration, attack_start_time=attack_start_time),
    'baseline-smc': CyberAttack(attack_type=attack_type, attack_intensity=attack_intensity, attack_frequency=attack_frequency, attack_duration=attack_duration, attack_start_time=attack_start_time),
    'baseline-pure-smc': CyberAttack(attack_type=attack_type, attack_intensity=attack_intensity, attack_frequency=attack_frequency, attack_duration=attack_duration, attack_start_time=attack_start_time),
    'omac-ood': CyberAttack(attack_type=attack_type, attack_intensity=attack_intensity, attack_frequency=attack_frequency, attack_duration=attack_duration, attack_start_time=attack_start_time),
    'omac-neuralFly': CyberAttack(attack_type=attack_type, attack_intensity=attack_intensity, attack_frequency=attack_frequency, attack_duration=attack_duration, attack_start_time=attack_start_time),
    'omac-deepRnnAtt': CyberAttack(attack_type=attack_type, attack_intensity=attack_intensity, attack_frequency=attack_frequency, attack_duration=attack_duration, attack_start_time=attack_start_time),
    'omac-pure-deepRnnAtt': CyberAttack(attack_type=attack_type, attack_intensity=attack_intensity, attack_frequency=attack_frequency, attack_duration=attack_duration, attack_start_time=attack_start_time),
    'omac-smc-deepRnnAtt': CyberAttack(attack_type=attack_type, attack_intensity=attack_intensity, attack_frequency=attack_frequency, attack_duration=attack_duration, attack_start_time=attack_start_time),
    'omac-pure-smc-deepRnnAtt': CyberAttack(attack_type=attack_type, attack_intensity=attack_intensity, attack_frequency=attack_frequency, attack_duration=attack_duration, attack_start_time=attack_start_time),
}

CONTROLLERS = {
    # 'baseline': controller.Baseline(),
    # 'baseline-pure': controller.BaselinePure(),
    # 'baseline-smc': controller.BaselineSmc(),
    # 'baseline-pure-smc': controller.BaselinePureSmc(),
    # 'omac-ood': controller.MetaAdaptOod(eta_a_base=eta_a, eta_A_base=eta_A_deep, dim_a=dim_a, layer_sizes=layer_sizes),
    # 'omac-neuralFly': controller.MetaAdaptNeuralFly(eta_a_base=eta_a, eta_A_base=eta_A_deep, dim_a=dim_a, layer_sizes=layer_sizes),
    # 'omac-deepRnnAtt': controller.MetaAdaptDeepRnnAtt(eta_a_base=eta_a, eta_A_base=eta_A_deep, dim_a=dim_a, layer_sizes=layer_sizes),
    # 'omac-pure-deepRnnAtt': controller.MetaAdaptPureDeepRnnAtt(eta_a_base=eta_a, eta_A_base=eta_A_deep, dim_a=dim_a, layer_sizes=layer_sizes),
    # 'omac-smc-deepRnnAtt': controller.MetaAdaptSmcDeepRnnAtt(eta_a_base=eta_a, eta_A_base=eta_A_deep, dim_a=dim_a, layer_sizes=layer_sizes),
    'omac-pure-smc-deepRnnAtt': controller.MetaAdaptPureSmcDeepRnnAtt(eta_a_base=eta_a, eta_A_base=eta_A_deep, dim_a=dim_a, layer_sizes=layer_sizes),
}

Train_Data = {}
Test_Data = {}
for ctrl in CONTROLLERS:
    Train_Data[ctrl] = []
    Test_Data[ctrl] = []

train_q_kwargs = {
    'Vwind' : np.array((0.0, 0, 0)),                                                # mean wind speed
    'VwindList' : np.random.gamma(shape=1., scale=1., size=(100,3)),                # predefined wind list
    'Vwind_gust' : np.array((5.0, 0., 2.5)),                                        # for hard wind constrant, wind speed is in the range Vwind +/- Vwind_gust
    'wind_model': 'predefined-list',                                                # {'random-walk', 'iid-uniform', 'predefined-list'}
    'wind_update_period' : 2.0,                                                     # seconds between wind speed changes
    't_stop' : time_stop,                                                           # total simulation time
}

TRAIN_QUADROTORS = {}
for ctrl in CONTROLLERS.keys():
    TRAIN_QUADROTORS[ctrl] = quadrotor.QuadrotorWithSideForceAndAttack(attacker=CYBERATTACKERS[ctrl],**train_q_kwargs)

for ctrl in CONTROLLERS:
    if hasattr(CONTROLLERS[ctrl], 'train'):
            CONTROLLERS[ctrl].train = True
    print('\nRunning training experiments for controller: %s' % ctrl)
    for i in range(num_train_experiments):
        print(' Train experiment %d/%d' % (i+1, num_train_experiments))
        print(f"{datetime.now()} Starting training experiment {i+1}/{num_train_experiments} for controller {ctrl}...")
        data = TRAIN_QUADROTORS[ctrl].run(trajectory=used_trajectory, controller=CONTROLLERS[ctrl], seed=seed+i)
        print(f"{datetime.now()} Finished training experiment {i+1}/{num_train_experiments} for controller {ctrl}.")
        Train_Data[ctrl].append(data)

np.random.seed(seed**2)
test_q_kwargs = {
    'Vwind' : np.array((0.0, 0, 0)),                                                # mean wind speed
    'VwindList' : np.random.uniform(low=-test_max_wind, high=0, size=(100,3)),      # predefined wind list
    'Vwind_gust' : np.array((5.0, 0., 2.5)),                                        # for hard wind constrant, wind speed is in the range Vwind +/- Vwind_gust
    'wind_model': 'predefined-list',                                                # {'random-walk', 'iid-uniform', 'predefined-list'}
    'wind_update_period' : 2.0,                                                     # seconds between wind speed changes
    't_stop' : time_stop,                                                           # total simulation time
}

TEST_QUADROTORS = {}
for ctrl in CONTROLLERS.keys():
    TEST_QUADROTORS[ctrl] = quadrotor.QuadrotorWithSideForceAndAttack(attacker=CYBERATTACKERS[ctrl], **test_q_kwargs)

for ctrl in CONTROLLERS:
    if hasattr(CONTROLLERS[ctrl], 'train'):
            CONTROLLERS[ctrl].train = False
    print('\nRunning test experiments for controller: %s' % ctrl)
    for i in range(num_test_experiments):
        print(' Test experiment %d/%d' % (i+1, num_test_experiments))
        print(f"{datetime.now()} Starting testing experiment {i+1}/{num_test_experiments} for controller {ctrl}...")
        data = TEST_QUADROTORS[ctrl].run(trajectory=used_trajectory, controller=CONTROLLERS[ctrl], seed=seed**2+i)
        print(f"{datetime.now()} Finished testing experiment {i+1}/{num_test_experiments} for controller {ctrl}.")
        Test_Data[ctrl].append(data)

# Save Train_Data to MATLAB format
print(f'\nSaving Train_Data to MATLAB format in {results_dir}...')
train_data_matlab = {}
for ctrl in Train_Data:
    ctrl_safe = matlab_safe_name(ctrl)
    for i, data in enumerate(Train_Data[ctrl]):
        # Create a unique key for each controller and experiment
        key = f'{ctrl_safe}_exp{i+1}'
        # Convert data dictionary to a format suitable for MATLAB
        train_data_matlab[key] = {}
        for field in data:
            # Convert numpy arrays and torch tensors to numpy arrays
            if isinstance(data[field], torch.Tensor):
                train_data_matlab[key][field] = data[field].cpu().numpy()
            elif isinstance(data[field], np.ndarray):
                train_data_matlab[key][field] = data[field]
            else:
                # For non-array data, try to convert to numpy array
                try:
                    train_data_matlab[key][field] = np.array(data[field])
                except:
                    # Skip fields that cannot be converted
                    pass

train_file = os.path.join(results_dir, f'{file_base}_Train.mat')
spio.savemat(train_file, train_data_matlab)
print(f'Train_Data saved to {train_file}')

# Save Test_Data to MATLAB format
print(f'\nSaving Test_Data to MATLAB format in {results_dir}...')
test_data_matlab = {}
for ctrl in Test_Data:
    ctrl_safe = matlab_safe_name(ctrl)
    for i, data in enumerate(Test_Data[ctrl]):
        # Create a unique key for each controller and experiment
        key = f'{ctrl_safe}_exp{i+1}'
        # Convert data dictionary to a format suitable for MATLAB
        test_data_matlab[key] = {}
        for field in data:
            # Convert numpy arrays and torch tensors to numpy arrays
            if isinstance(data[field], torch.Tensor):
                test_data_matlab[key][field] = data[field].cpu().numpy()
            elif isinstance(data[field], np.ndarray):
                test_data_matlab[key][field] = data[field]
            else:
                # For non-array data, try to convert to numpy array
                try:
                    test_data_matlab[key][field] = np.array(data[field])
                except:
                    # Skip fields that cannot be converted
                    pass

test_file = os.path.join(results_dir, f'{file_base}_Test.mat')
spio.savemat(test_file, test_data_matlab)
print(f'Test_Data saved to {test_file}')

# Save statistics to MATLAB format
print(f'\nSaving statistics to MATLAB format in {results_dir}...')
stats_matlab = {
    'train_stats_position': {},
    'test_stats_position': {},
    'train_stats_ace': {},
    'test_stats_ace': {}
}
