import numpy as np


class CyberAttack:
    def __init__(self, attack_type='none', attack_intensity=0.1, attack_frequency=0.1, attack_duration=1.0, attack_start_time=5.0):
        
        ##---------- Attack parameters ----------##
        self.attack_type = attack_type                      # attack type: 'none', 'sensor_bias', 'actuator_bias', 'sensor_false_data_injection', 'actuator_false_data_injection', 'sensor_dos', 'actuator_dos', 'sensor_delay', 'actuator_delay'
        self.attack_intensity = attack_intensity            # attack intensity
        self.attack_frequency = attack_frequency            # attack frequency
        self.attack_duration = attack_duration              # attack duration
        self.attack_start_time = attack_start_time          # attack start time
        
        ##---------- Attack state ----------##
        self.attack_active = False                          # whether attack is active
        self.attack_timer = 0.0                             # timer for current attack
        self.last_attack_time = 0.0                         # last attack time

        ##---------- Quadrotor attack related ----------##
        self.quadrotor_amplification_factor = 150.0         # amplification factor for quadrotor attacks
        self.quadrotor_false_data_injection_intensity = 0.2 # threshold for quadrotor DoS attack success
        
        ##---------- DoS attack related ----------##
        self.sensor_dos_buffer = []                         # buffer for sensor DoS attacks
        self.max_sensor_dos_buffer_size = 10                # maximum buffer size for sensor DoS attacks
        self.sensor_dos_try_attack_counter = 0              # counter for sensor DoS attacks
        self.sensor_dos_real_attack_counter = 0             # counter for sensor successful DoS attacks
        self.sensor_last_valid_signal = None                # last valid signal for sensor DoS attacks
        self.actuator_dos_buffer = []                       # buffer for actuator DoS attacks
        self.max_actuator_dos_buffer_size = 10              # maximum buffer size for actuator DoS attacks
        self.actuator_dos_try_attack_counter = 0            # counter for actuator DoS attacks
        self.actuator_dos_real_attack_counter = 0           # counter for actuator successful DoS attacks
        self.actuator_last_valid_signal = None              # last valid signal for actuator DoS attacks
        self.dos_attack_intensity = attack_intensity * 9    # probability of successful DoS attack
        
        ##---------- Delay attack related ----------##
        self.sensor_delay_buffer = []                       # buffer for sensor delayed signals
        self.actuator_delay_buffer = []                     # buffer for actuator delayed signals
        self.max_delay_steps = 50                           # maximum delay steps

        ##---------- Local random number generator ----------##
        self._local_rng = np.random.default_rng(51**1)                 # local random number generator for DoS attack
        self._local_rng_sensor_dos = np.random.default_rng(51**2)      # local random number generator for delay attack
        self._local_rng_actuator_dos = np.random.default_rng(51**3)    # local random number generator for delay attack

    # Update attack status based on current time
    def update_attack_status(self, current_time, step_size):
        
        # No attack
        if self.attack_type == 'none':
            return # no attack
        
        # Check if attack should start
        if current_time < self.attack_start_time:
            self.attack_active = False
            return # no attack before start time
        
        # Determine if attack starts or ends
        if not self.attack_active:
            if self._local_rng.random() < self.attack_frequency * step_size:
                self.attack_active = True
                self.attack_timer = 0.0
                self.last_attack_time = current_time
        else:
            self.attack_timer += step_size
            if self.attack_timer >= self.attack_duration:
                self.attack_active = False
                self.attack_timer = 0.0
    
    # Apply sensor attack
    def apply_sensor_attack(self, state):
        if not self.attack_active or self.attack_type != 'sensor_bias':
            return state
        attack_state = np.array([
            self.attack_intensity * np.sin(2 * np.pi * self.attack_timer), 
            self.attack_intensity * np.cos(2 * np.pi * self.attack_timer),
            0.0])
        return state + attack_state
    
    # Apply actuator attack
    def apply_actuator_attack(self, control_signal):
        if not self.attack_active or self.attack_type != 'actuator_bias':
            return control_signal
        attack_signal = np.array([
            self.attack_intensity * np.sin(1 * np.pi * self.attack_timer) * self.quadrotor_amplification_factor,
            self.attack_intensity * np.cos(1 * np.pi * self.attack_timer) * self.quadrotor_amplification_factor,
            self.attack_intensity * np.sin(2 * np.pi * self.attack_timer) * self.quadrotor_amplification_factor,
            self.attack_intensity * np.cos(2 * np.pi * self.attack_timer) * self.quadrotor_amplification_factor
        ])
        return control_signal + attack_signal
    
    # Apply sensor false data injection attack
    def apply_sensor_false_data_injection(self, measurement, true_value):
        if not self.attack_active or self.attack_type != 'sensor_false_data_injection':
            return measurement
        false_component = self.attack_intensity * true_value * np.sin(2 * np.pi * self.attack_timer) * self.quadrotor_false_data_injection_intensity
        return measurement + false_component
    
    # Apply actuator false data injection attack
    def apply_actuator_false_data_injection(self, measurement, true_value):
        if not self.attack_active or self.attack_type != 'actuator_false_data_injection':
            return measurement
        false_component = self.attack_intensity * true_value * np.sin(2 * np.pi * self.attack_timer) * self.quadrotor_false_data_injection_intensity * self.quadrotor_amplification_factor
        return measurement + false_component
    
    # Apply sensor DoS attack
    def apply_sensor_dos_attack(self, signal):
        self.sensor_dos_buffer.append(signal)
        if len(self.sensor_dos_buffer) > self.max_sensor_dos_buffer_size:
            self.sensor_dos_buffer.pop(0)
        if not self.attack_active or self.attack_type != 'sensor_dos':
            self.sensor_last_valid_signal = signal
            return signal, True
        self.sensor_dos_try_attack_counter += 1
        if self._local_rng.random() < self.dos_attack_intensity:
            self.sensor_dos_real_attack_counter += 1
            if hasattr(self, 'sensor_last_valid_signal') and self.sensor_last_valid_signal is not None:
                return self.sensor_last_valid_signal, False
            else:
                return np.zeros_like(signal), False
        self.sensor_last_valid_signal = signal
        return signal, True
    
    # Apply actuator DoS attack
    def apply_actuator_dos_attack(self, signal):
        self.actuator_dos_buffer.append(signal)
        if len(self.actuator_dos_buffer) > self.max_actuator_dos_buffer_size:
            self.actuator_dos_buffer.pop(0)
        if not self.attack_active or self.attack_type != 'actuator_dos':
            self.actuator_last_valid_signal = signal
            return signal, True
        self.actuator_dos_try_attack_counter += 1
        if self._local_rng_actuator_dos.random() < self.dos_attack_intensity:
            self.actuator_dos_real_attack_counter += 1
            if hasattr(self, 'actuator_last_valid_signal') and self.actuator_last_valid_signal is not None:
                return self.actuator_last_valid_signal, False
            else:
                return np.zeros_like(signal), False
        self.actuator_last_valid_signal = signal
        return signal, True
    
    # Apply delay attack
    def apply_sensor_delay_attack(self, signal):
        self.sensor_delay_buffer.append(signal)
        if len(self.sensor_delay_buffer) > self.max_delay_steps:
            self.sensor_delay_buffer.pop(0)
        if not self.attack_active or self.attack_type != 'sensor_delay':
            return signal
        if len(self.sensor_delay_buffer) > 10:
            delay_steps = min(int(self.attack_intensity * self.max_delay_steps / 2), len(self.sensor_delay_buffer) - 1)
            return self.sensor_delay_buffer[-delay_steps-5]
    
    # Apply delay attack
    def apply_actuator_delay_attack(self, signal):
        self.actuator_delay_buffer.append(signal)
        if len(self.actuator_delay_buffer) > self.max_delay_steps:
            self.actuator_delay_buffer.pop(0)
        if not self.attack_active or self.attack_type != 'actuator_delay':
            return signal
        if len(self.actuator_delay_buffer) > 10:
            delay_steps = min(int(self.attack_intensity * self.max_delay_steps / 2), len(self.actuator_delay_buffer) - 1)
            return self.actuator_delay_buffer[-delay_steps-5]

    # Get current attack information
    def get_attack_info(self):
        return {
            'active'    : self.attack_active,
            'type'      : self.attack_type,
            'intensity' : self.attack_intensity,
            'timer'     : self.attack_timer
        }
