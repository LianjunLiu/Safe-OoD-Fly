import os
import rowan
import torch
import collections
import numpy as np
import torch.nn as nn
from warnings import warn
import torch.optim as optim
from utils import readparamfile
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


torch.set_default_dtype(torch.float64)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PX4_PARAM_FILE = os.path.join(_SCRIPT_DIR, 'params', 'px4.json')
DEFAULT_CONTROL_PARAM_FILE = os.path.join(_SCRIPT_DIR, 'params', 'controller.json')
DEFAULT_QUAD_PARAMETER_FILE = os.path.join(_SCRIPT_DIR, 'params', 'quadrotor.json')

# Define a named tuple for force so that subclasses can also pass around derivatives of force
Force = collections.namedtuple('Force', 'F F_dot F_ddot', defaults=(np.zeros(3), np.zeros(3)))


class Controller():
    ''' Controller class implements the attitude controller and thrust mixing. The position controller is not implemented and should be implemented in child classes'''
    
    _name = None
    name_long = None
    
    def __init__(self, quadparamfile=DEFAULT_QUAD_PARAMETER_FILE, px4paramfile=DEFAULT_PX4_PARAM_FILE):
        self.params = readparamfile(quadparamfile)
        self.px4_params = readparamfile(px4paramfile) 
        self.px4_params['angrate_max'] = np.array((self.px4_params['MC_ROLLRATE_MAX'], self.px4_params['MC_PITCHRATE_MAX'], self.px4_params['MC_YAWRATE_MAX']))
        self.px4_params['angrate_gain_P'] = np.diag((self.px4_params['MC_ROLLRATE_P'], self.px4_params['MC_PITCHRATE_P'], self.px4_params['MC_YAWRATE_P']))
        self.px4_params['angrate_gain_I'] = np.diag((self.px4_params['MC_ROLLRATE_I'], self.px4_params['MC_PITCHRATE_I'], self.px4_params['MC_YAWRATE_I']))
        self.px4_params['angrate_gain_D'] = np.diag((self.px4_params['MC_ROLLRATE_D'], self.px4_params['MC_PITCHRATE_D'], self.px4_params['MC_YAWRATE_D']))
        self.px4_params['angrate_gain_K'] = np.diag((self.px4_params['MC_ROLLRATE_K'], self.px4_params['MC_PITCHRATE_K'], self.px4_params['MC_YAWRATE_K']))
        self.px4_params['angrate_int_lim'] = np.array((self.px4_params['MC_RR_INT_LIM'], self.px4_params['MC_PR_INT_LIM'], self.px4_params['MC_YR_INT_LIM']))
        self.px4_params['attitude_gain_P'] = np.diag((self.px4_params['MC_ROLL_P'], self.px4_params['MC_PITCH_P'], self.px4_params['MC_YAW_P']))
        self.px4_params['angacc_max'] = np.array(self.px4_params['angacc_max'])
        self.B = None
    
    def reset_controller(self):
        # Reset Angular rate control parameters
        self.w_error_integral = np.zeros(3)
        self.w_filtered = np.zeros(3)
        self.w_filtered_last = np.zeros(3)
    
    def position(self, X, imu=np.zeros(3), pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), sd=np.zeros(3), wind=None, logentry=None):
        
        raise NotImplementedError

    def limit(self, array, upper_limit, lower_limit=None):
        """ 
        Ensure upper_limit >= array >= lower_limit, array must be mutable and is modified in place
        """
        
        if lower_limit is None:
            lower_limit = - upper_limit
        array[array > upper_limit] = upper_limit[array > upper_limit]
        array[array < lower_limit] = lower_limit[array < lower_limit]
    
    def attitude(self, q, q_sp, w_d=np.zeros(3)):
        """
        MC Attitude Controller for PX4
        https://dev.px4.io/master/en/flight_stack/controller_diagrams.html
        """

        q_error = rowan.multiply(rowan.inverse(q), q_sp)
        omega_sp = 2 * self.px4_params['attitude_gain_P'] @ (np.sign(q_error[0]) * q_error[1:])
        
        # omega_sp += w_d
        self.limit(omega_sp, self.px4_params['angrate_max'])

        return omega_sp

    def angrate(self, w, w_sp, dt, logentry):
        """
        Angular rate controller that matches PX4 implementation here:
        https://docs.px4.io/master/en/config_mc/pid_tuning_guide_multicopter.html#rate-controller
        """
        
        # Calculate the angular rate error
        w_error = w_sp - w

        # Integrate the error signal and limit it
        self.w_error_integral += dt * w_error
        self.limit(self.w_error_integral, self.px4_params['angrate_int_lim'])
        
        # Sanity check the limit function since you never bothered to write a unit test for it
        if any(self.w_error_integral > self.px4_params['angrate_int_lim']) or any(self.w_error_integral < -self.px4_params['angrate_int_lim']) :
            raise ValueError

        # Calculate the derivative of the filtered angular rate PX4 does not include derivative of setpoint
        const_w_filter = np.exp(- dt / self.px4_params['w_filter_time_const'])
        self.w_filtered *= const_w_filter
        self.w_filtered += (1 - const_w_filter) * w
        
        w_filtered_derivative = (self.w_filtered - self.w_filtered_last) / dt
        logentry['w_filtered_last'] = self.w_filtered_last.copy()
        self.w_filtered_last[:] = self.w_filtered[:]

        alpha_sp = self.px4_params['angrate_gain_K'] @ (self.px4_params['angrate_gain_P'] @ w_error + self.px4_params['angrate_gain_I'] @ self.w_error_integral - self.px4_params['angrate_gain_D'] @ w_filtered_derivative)

        self.limit(alpha_sp, self.px4_params['angacc_max'])
        torque_sp = alpha_sp

        logentry['w'] = w
        logentry['w_error_integral'] = self.w_error_integral
        logentry['w_filtered'] = self.w_filtered
        logentry['w_filtered_derivative'] = w_filtered_derivative
        logentry['alpha_sp'] = alpha_sp

        return torque_sp

    def mixer_get_motorspeedsquared(self, torque_sp, T_sp):
        
        return np.linalg.solve(self.B, np.concatenate(((T_sp,), torque_sp)))

    def mixer(self, torque_sp, T_sp, logentry):
        """ 
        Calculate motor speed commands from torque and thrust set points 
            - Assumes T_sp > 0
            - ~~reduces torque_sp[2] first if max and min motor speeds violated~~
                ^ incomplete and commented out
        """

        omega_squared = np.linalg.solve(self.B, np.concatenate(((T_sp,), torque_sp)))
        omega = np.sqrt(np.maximum(omega_squared, self.params['motor_min_speed']))
        omega = np.minimum(omega, self.params['motor_max_speed'])

        logentry['motor_speed_command'] = omega

        return omega

    def calculate_derivative(self, x, x_last, dt_inv):
        """
        Calculate the x and handles the case where x_last or dt_inv are initialized to None
        """ 
        
        try:
            x_dot = (x - x_last) * dt_inv
            x_last = x.copy()
        except TypeError as err:
            if x_last is None or dt_inv is None:
                x_dot = np.zeros_like(x)
                x_last = x.copy()
                return x_dot, x_last
            else:
                raise err
        
        return x_dot, x_last


class Baseline(Controller):
    ''' Baseline controller class with PID feedback and acceleration feedforward term and with exact quadrotor dynamic model'''
    
    _name = 'baseline-pid'
    name_long = 'baseline-pid'

    def __init__(self, ctrlparamfile=DEFAULT_CONTROL_PARAM_FILE, quadparamfile=DEFAULT_QUAD_PARAMETER_FILE, integral_control=True, **kwargs):
        super().__init__(quadparamfile=quadparamfile, **kwargs)
        self.params = readparamfile(filename=ctrlparamfile, params=self.params)
        self.integral_control = integral_control

        self.e = np.zeros(3)
        self.dot_e = np.zeros(3)
        self.int_e = np.zeros(3)
        self.e_p = np.zeros(3)
        self.dot_e_p = np.zeros(3)
        self.int_e_p = np.zeros(3)
        self.integral_limit = 2.0
        self.x_pred = np.zeros(3)
    
    def calculate_gains(self):
        self.params['K_p'] = np.diag(
            [
                self.params['Lam_xy'] * self.params['K_xy'],
                self.params['Lam_xy'] * self.params['K_xy'],
                self.params['Lam_z'] * self.params['K_z']
            ]
        )
        self.params['K_i'] = np.array(self.params['K_i'])
        if not self.integral_control:
            self.params['K_i'] = np.zeros((3, 3))
        self.params['K_d'] = np.diag(
            [
                self.params['K_xy'], 
                self.params['K_xy'], 
                self.params['K_z']
            ]
        )
        self.B = np.array(
            [
                self.params['C_T'] * np.array([1., 1., 1., 1.]), 
                self.params['C_T'] * self.params['l_arm'] * np.array([-1., -1., 1., 1.]),
                self.params['C_T'] * self.params['l_arm'] * np.array([-1., 1., 1., -1.]),
                self.params['C_q'] * np.array([-1., 1., -1., 1.])
            ]
        )
    
    def reset_controller(self):
        super().reset_controller()
        self.calculate_gains()
        self.F_r_dot = None
        self.F_r_last = None
        self.t_last = None
        self.t_last_wind_update = None

        self.p_error = np.zeros(3)
        self.d_error = np.zeros(3)
        self.i_error = np.zeros(3)

        self.dt = 0.
        self.dt_inv = 0.

        self.e = np.zeros(3)
        self.dot_e = np.zeros(3)
        self.int_e = np.zeros(3)
        self.e_p = np.zeros(3)
        self.dot_e_p = np.zeros(3)
        self.int_e_p = np.zeros(3)
        self.integral_limit = 2.0
        self.x_pred = np.zeros(3)
    
    def get_q(self, F_r, yaw_desired, max_angle=np.pi, check=False):
        """
        Finds quaterion that will give rotation to align body z-axis with with F_r and yield with desired yaw
        """

        q_world_to_yawed = rowan.from_euler(0., 0., yaw_desired, 'xyz')

        rotation_axis = np.cross((0, 0, 1), F_r)
        if np.allclose(rotation_axis, (0., 0., 0.)):
            unit_rotation_axis = np.array((1., 0., 0.,))
        else:
            unit_rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_axis /= np.linalg.norm(F_r)
        rotation_angle = np.arcsin(np.linalg.norm(rotation_axis))
        if F_r[2] < 0:
            rotation_angle = np.pi - rotation_angle
        if rotation_angle > max_angle:
            rotation_angle = max_angle
        q_yawed_to_body = rowan.from_axis_angle(unit_rotation_axis, rotation_angle)

        q_r = rowan.multiply(q_world_to_yawed, q_yawed_to_body)

        if any(np.isnan(q_r)):
            raise TypeError
        
        return q_r

    def get_Fr(self, X, imu=np.zeros(3), pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), meta_adapt_trigger=None, wind=None, attack_active=None, logentry=None, **kwargs):
        
        residual = np.zeros(3)
        p_purified = np.zeros(3)
        self.x_pred = np.zeros(3)

        self.e = - pd + X[0:3]
        self.dot_e = - vd + X[7:10]
        self.int_e += self.dt * self.e
        self.int_e = np.clip(self.int_e, -self.integral_limit, self.integral_limit)

        self.e_p = -pd + p_purified
        self.dot_e_p = - vd + X[7:10]
        self.int_e_p += self.dt * self.e_p
        self.int_e_p = np.clip(self.int_e_p, -self.integral_limit, self.integral_limit)
        
        self.p_error = self.e
        self.d_error = self.dot_e
        self.i_error += self.dt * self.e

        a_r = - self.params['K_p'] @ self.p_error - self.params['K_d'] @ self.d_error - self.params['K_i'] @ self.i_error + ad
        F_r = (a_r * self.params['m']) + np.array([0., 0., self.params['m'] * self.params['g']])

        try:
            lam = np.exp(- self.dt / self.params['force_filter_time_const'])
            self.F_r_dot *= lam
            self.F_r_dot += (1 - lam) * (F_r - self.F_r_last) * self.dt_inv
        except TypeError as err:
            if self.F_r_last is None:
                self.F_r_dot = np.zeros(3)
            else:
                raise err

        if any(np.isinf(self.F_r_dot)):
            raise ValueError
        self.F_r_last = F_r.copy()

        if logentry is not None:
            logentry['residual'] = residual
            logentry['p_purified'] = p_purified
            logentry['x_pred'] = self.x_pred
            logentry['e'] = self.e
            logentry['dot_e'] = self.dot_e
            logentry['int_e'] = self.int_e
            logentry['e_p'] = self.e_p
            logentry['dot_e_p'] = self.dot_e_p
            logentry['int_e_p'] = self.int_e_p
            logentry['p_error'] = self.p_error
            logentry['d_error'] = self.d_error
            logentry['i_error'] = self.i_error
            logentry['p_term'] = - self.params['K_p'] @ self.p_error * self.params['m']
            logentry['d_term'] = - self.params['K_d'] @ self.d_error * self.params['m']
            logentry['i_term'] = - self.params['K_i'] @ self.i_error * self.params['m']
            logentry['ad_term'] = ad * self.params['m']
            logentry['jd_term'] = jd * self.params['m']
            logentry['g_term'] = np.array([0., 0., self.params['m'] * self.params['g']])
            
        return Force(F_r, self.F_r_dot)

    def position(self, X, imu=np.zeros(3), pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), sd=np.zeros(3), t=None, wind=None, attack_active=None, logentry=None, t_last_wind_update=None):
        try:
            self.dt = t - self.t_last
            self.dt_inv = 1 / self.dt
        except TypeError as err:
            if self.t_last is None or t is None:
                pass
            else:
                raise err
        try:
            if self.t_last_wind_update < t_last_wind_update:
                self.t_last_wind_update = t_last_wind_update
                meta_adapt_trigger = True
            else:
                meta_adapt_trigger = False
        except TypeError as err:
            self.t_last_wind_update = t_last_wind_update
            meta_adapt_trigger = False

        self.t_last = t
        force = self.get_Fr(X, imu=imu, pd=pd, vd=vd, ad=ad, jd=jd, meta_adapt_trigger=meta_adapt_trigger, wind=wind, attack_active=attack_active, logentry=logentry)
        F_r = force.F  # *_r is the force command
        
        yaw_d = 0.0

        # Compute thrust and quaternion with first order delay compensation
        F_r_dot = force.F_dot
        T_r_prime = np.linalg.norm(F_r + self.params['thrust_delay'] * F_r_dot)
        q_r_prime = self.get_q(F_r + self.params['attitude_delay'] * F_r_dot, yaw_d)
        F_r_prime = rowan.to_matrix(q_r_prime) @ np.array((0, 0, T_r_prime))

        # The following two blocks of code limit the commanded force to be within the maximum thrust and maximum attitude limits
        # Find projection of thrust onto cone within max_zenith_angle of zenith
        T_max = self.params['max_thrust']
        T_hover = self.params['m'] * self.params['g']
        if np.isnan(self.params['max_zenith_angle']):
            s_oncone = float('inf')
        elif self.params['max_zenith_angle'] >= np.pi - 1e-2:
            s_oncone = float('inf')
        elif np.sqrt(F_r_prime[0]**2 + F_r_prime[1]**2) < 1e-4:
            s_oncone = float('inf')
        elif F_r_prime[2] / np.sqrt(F_r_prime[0]**2 + F_r_prime[1]**2) >= (np.cos(self.params['max_zenith_angle'])/np.sin(self.params['max_zenith_angle'])):
            s_oncone = float('inf')
        else:
            s_oncone = T_hover / ((np.cos(self.params['max_zenith_angle'])/np.sin(self.params['max_zenith_angle'])) * np.sqrt(F_r_prime[0]**2 + F_r_prime[1]**2) - (F_r_prime[2] - T_hover))
        # Find projection of thrust onto sphere of max_thrust
        a = np.linalg.norm(F_r_prime - np.array([0., 0., T_hover])) ** 2
        b = 2 * T_hover * (F_r_prime[2] - T_hover)
        c = T_hover ** 2 - T_max ** 2
        if a >= 1e-4:
            s_onsphere = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        else:
            s_onsphere = float('inf')

        s = min(1, s_onsphere, s_oncone)
        F_r_prime = s * (F_r_prime - np.array([0., 0., T_hover])) + np.array([0., 0., T_hover])
        
        T_r_prime = np.linalg.norm(F_r_prime)
        q_r_prime = self.get_q(F_r_prime, yaw_d)

        if np.isnan(q_r_prime[0]):
            raise ValueError('quaternion was nan, maybe F_r[2] was 0 and the drone is trying to flip??')
        if T_r_prime > 124.:
            warn('thrust gets too high')
        
        T_d = self.params['m'] * np.linalg.norm(ad)
        q_d = self.get_q(ad + np.array((0., 0., self.params['g'])), yaw_d, check=True)

        if logentry is not None:
            logentry['q_d'] = q_d
            logentry['T_d'] = T_d
            logentry['F_r'] = F_r
            logentry['F_r_dot'] = F_r_dot
            logentry['F_r_prime'] = F_r_prime
            logentry['s_oncone'] = s_oncone
            logentry['s_onsphere'] = s_onsphere
            logentry['meta_adapt_trigger'] = meta_adapt_trigger
        
        return T_r_prime, q_r_prime


class BaselinePure(Baseline):

    _name = 'baseline-pure'
    name_long = 'baseline-pure'

    def __init__(self, integral_control=True, *args, **kwargs):
        super().__init__(integral_control=integral_control, *args, **kwargs)
    
    def get_Fr(self, X, imu=np.zeros(3), pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), meta_adapt_trigger=None, wind=None, attack_active=None, logentry=None, **kwargs):
        
        residual = np.linalg.norm(X[0:3] - self.x_pred[0:3])
        sigma = 0.02 
        weight = np.exp(- (residual**2) / (2 * sigma**2))
        p_purified = weight * X[0:3] + (1 - weight) * self.x_pred[0:3]
        self.x_pred[0:3] = p_purified + X[7:10] * self.dt

        self.e = - pd + X[0:3]
        self.dot_e = - vd + X[7:10]
        self.int_e += self.dt * self.e
        self.int_e = np.clip(self.int_e, -self.integral_limit, self.integral_limit)

        self.e_p = -pd + p_purified
        self.dot_e_p = - vd + X[7:10]
        self.int_e_p += self.dt * self.e_p
        self.int_e_p = np.clip(self.int_e_p, -self.integral_limit, self.integral_limit)
        
        self.p_error = self.e_p
        self.d_error = self.dot_e_p
        self.i_error += self.dt * self.e_p

        a_r = - self.params['K_p'] @ self.p_error - self.params['K_d'] @ self.d_error - self.params['K_i'] @ self.i_error + ad
        F_r = (a_r * self.params['m']) + np.array([0., 0., self.params['m'] * self.params['g']])

        try:
            lam = np.exp(- self.dt / self.params['force_filter_time_const'])
            self.F_r_dot *= lam
            self.F_r_dot += (1 - lam) * (F_r - self.F_r_last) * self.dt_inv
        except TypeError as err:
            if self.F_r_last is None:
                self.F_r_dot = np.zeros(3)
            else:
                raise err

        if any(np.isinf(self.F_r_dot)):
            raise ValueError
        self.F_r_last = F_r.copy()

        if logentry is not None:
            logentry['residual'] = residual
            logentry['p_purified'] = p_purified
            logentry['x_pred'] = self.x_pred
            logentry['e'] = self.e
            logentry['dot_e'] = self.dot_e
            logentry['int_e'] = self.int_e
            logentry['e_p'] = self.e_p
            logentry['dot_e_p'] = self.dot_e_p
            logentry['int_e_p'] = self.int_e_p
            logentry['p_error'] = self.p_error
            logentry['d_error'] = self.d_error
            logentry['i_error'] = self.i_error
            logentry['p_term'] = - self.params['K_p'] @ self.p_error * self.params['m']
            logentry['d_term'] = - self.params['K_d'] @ self.d_error * self.params['m']
            logentry['i_term'] = - self.params['K_i'] @ self.i_error * self.params['m']
            logentry['ad_term'] = ad * self.params['m']
            logentry['jd_term'] = jd * self.params['m']
            logentry['g_term'] = np.array([0., 0., self.params['m'] * self.params['g']])
            
        return Force(F_r, self.F_r_dot)

class BaselineSmc(Baseline):

    _name = 'baseline-smc'
    name_long = 'baseline-smc'

    def __init__(self, integral_control=True, *args, **kwargs):
        super().__init__(integral_control=integral_control, *args, **kwargs)

        def _to_diag(val):
            arr = np.array(val, dtype=float)
            if arr.ndim == 0:
                arr = np.full(3, arr)
            elif arr.size != 3:
                raise ValueError('expected scalar or 3-vector')
            return np.diag(arr)
        
        self.integral_limit = 2.0
        self.alpha_s = _to_diag(1.2)
        self.beta_s = _to_diag(0.6)
        self.lam_s = _to_diag(0.3)
        self.a_s = 1.50
        self.b_s = 1.20
        self.K_s = _to_diag(3.5)
        self.K_sw = _to_diag(2.5)
        self.phi_sw = float(0.5)
    
    def get_Fr(self, X, imu=np.zeros(3), pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), meta_adapt_trigger=None, wind=None, attack_active=None, logentry=None, **kwargs):
        
        residual = np.zeros(3)
        p_purified = np.zeros(3)
        self.x_pred = np.zeros(3)

        self.e = - pd + X[0:3]
        self.dot_e = - vd + X[7:10]
        self.int_e += self.dt * self.e
        self.int_e = np.clip(self.int_e, -self.integral_limit, self.integral_limit)

        self.e_p = -pd + p_purified
        self.dot_e_p = - vd + X[7:10]
        self.int_e_p += self.dt * self.e_p
        self.int_e_p = np.clip(self.int_e_p, -self.integral_limit, self.integral_limit)

        self.p_error = self.e
        self.d_error = self.dot_e
        self.i_error = self.int_e

        ##---------- 计算滑模控制 ----------##
        # 1. 计算滑模面 s （s = e + alpha * |e|^a * sgn(e) + beta * |v_e|^b * sgn(v_e) + lam * integral）
        term_p = self.alpha_s @ (np.abs(self.p_error)**self.a_s * np.sign(self.p_error))
        term_d = self.beta_s @ (np.abs(self.d_error)**self.b_s * np.sign(self.d_error))
        term_i = self.lam_s @ self.i_error
        s = self.p_error + term_p + term_d + term_i

        # 2. 计算趋近律项 (Reach Law) （增加 phi_sw 边界层厚度至 1.5-2.5 以增强对攻击跳变的鲁棒性）
        sat_s = np.tanh(s / self.phi_sw)
        dot_s_target = -self.K_s @ s - self.K_sw @ sat_s

        # 3. 计算等效控制需要的中间项 （鲁棒化处理：增大 epsilon 防止分母过小导致控制量因攻击数据剧震）
        epsilon = 0.05
        # coeff_beta = beta * b * |v_e|^(b-1)
        coeff_beta = self.beta_s @ np.diag(self.b_s * (np.abs(self.d_error)**(self.b_s - 1) + epsilon))
        # term_alpha = alpha * a * |e|^(a-1) * v_e
        term_alpha = self.alpha_s @ np.diag(self.a_s * (np.abs(self.p_error)**(self.a_s - 1))) @ self.d_error
        # term_lam = lam * e
        term_lam = self.lam_s @ self.p_error

        # 4. 解出 a_smc （公式推导：dot{s} = v_e + term_alpha + coeff_beta @ (a_smc - ad) + term_lam = dot_s_target）（整理得：a_smc = ad + inv(coeff_beta) @ (dot_s_target - v_error - term_alpha - term_lam)）
        inv_coeff_beta = np.linalg.inv(coeff_beta)
        # 鲁棒化处理：限制逆矩阵增益的最大值，防止受攻击时产生天文数字般的加速度
        inv_coeff_beta = np.clip(inv_coeff_beta, -5.0, 5.0)
        a_smc = ad + inv_coeff_beta @ (dot_s_target - self.d_error - term_alpha - term_lam)
        
        gravity = np.array([0., 0., -self.params['g']])
        F_r = self.params['m'] * (a_smc - gravity)
        
        try:
            lam = np.exp(- self.dt / self.params['force_filter_time_const'])
            self.F_r_dot *= lam
            self.F_r_dot += (1 - lam) * (F_r - self.F_r_last) * self.dt_inv
        except TypeError as err:
            if self.F_r_last is None:
                self.F_r_dot = np.zeros(3)
            else:
                raise err
        
        if any(np.isinf(self.F_r_dot)):
            raise ValueError
        self.F_r_last = F_r.copy()
        
        if logentry is not None:
            logentry['residual'] = residual
            logentry['p_purified'] = p_purified
            logentry['x_pred'] = self.x_pred
            logentry['e'] = self.e
            logentry['dot_e'] = self.dot_e
            logentry['int_e'] = self.int_e
            logentry['e_p'] = self.e_p
            logentry['dot_e_p'] = self.dot_e_p
            logentry['int_e_p'] = self.int_e_p
            logentry['p_error'] = self.p_error
            logentry['d_error'] = self.d_error
            logentry['i_error'] = self.i_error
            logentry['term_p'] = term_p
            logentry['term_v'] = term_d
            logentry['term_i'] = term_i
            logentry['s'] = s
            logentry['sat_s'] = sat_s
            logentry['dot_s_target'] = dot_s_target
            logentry['coeff_beta'] = coeff_beta
            logentry['term_alpha'] = term_alpha
            logentry['term_lam'] = term_lam
            logentry['inv_coeff_beta'] = inv_coeff_beta
            logentry['a_smc'] = a_smc
            logentry['ad_term'] = ad * self.params['m']
            logentry['jd_term'] = jd * self.params['m']
            logentry['g_term'] = np.array([0., 0., self.params['m'] * self.params['g']])

        return Force(F_r, self.F_r_dot)


class BaselinePureSmc(Baseline):

    _name = 'baseline-pure-smc'
    name_long = 'baseline-pure-smc'

    def __init__(self, integral_control=True, *args, **kwargs):
        super().__init__(integral_control=integral_control, *args, **kwargs)

        def _to_diag(val):
            arr = np.array(val, dtype=float)
            if arr.ndim == 0:
                arr = np.full(3, arr)
            elif arr.size != 3:
                raise ValueError('expected scalar or 3-vector')
            return np.diag(arr)
        
        self.integral_limit = 2.0
        self.alpha_s = _to_diag(1.2)
        self.beta_s = _to_diag(0.6)
        self.lam_s = _to_diag(0.3)
        self.a_s = 1.50
        self.b_s = 1.20
        self.K_s = _to_diag(3.5)
        self.K_sw = _to_diag(2.5)
        self.phi_sw = float(0.5)
    
    def get_Fr(self, X, imu=np.zeros(3), pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), meta_adapt_trigger=None, wind=None, attack_active=None, logentry=None, **kwargs):
        
        ##---------- 物理预测值与测量值的软融合 ----------##
        # 1. 计算测量值与物理预测值的残差 （假设 self.x_pred 是利用名义动力学推算的上一时刻状态）
        residual = np.linalg.norm(X[0:3] - self.x_pred[0:3])
        # 2. 计算置信度权重 (sigma 可调，决定对偏差的容忍度)
        sigma = 0.02 
        weight = np.exp(- (residual**2) / (2 * sigma**2))
        # 3. 软融合：攻击越大，weight 越小，越依赖预测值
        p_purified = weight * X[0:3] + (1 - weight) * self.x_pred[0:3]
        # 4. 更新物理预测值供下一帧使用 (简化的运动学)
        self.x_pred[0:3] = p_purified + X[7:10] * self.dt

        self.e = - pd + X[0:3]
        self.dot_e = - vd + X[7:10]
        self.int_e += self.dt * self.e
        self.int_e = np.clip(self.int_e, -self.integral_limit, self.integral_limit)

        self.e_p = -pd + p_purified
        self.dot_e_p = - vd + X[7:10]
        self.int_e_p += self.dt * self.e_p
        self.int_e_p = np.clip(self.int_e_p, -self.integral_limit, self.integral_limit)

        self.p_error = self.e_p
        self.d_error = self.dot_e_p
        self.i_error = self.int_e_p

        ##---------- 计算滑模控制 ----------##
        # 1. 计算滑模面 s （s = e + alpha * |e|^a * sgn(e) + beta * |v_e|^b * sgn(v_e) + lam * integral）
        term_p = self.alpha_s @ (np.abs(self.p_error)**self.a_s * np.sign(self.p_error))
        term_d = self.beta_s @ (np.abs(self.d_error)**self.b_s * np.sign(self.d_error))
        term_i = self.lam_s @ self.i_error
        s = self.p_error + term_p + term_d + term_i

        # 2. 计算趋近律项 (Reach Law) （增加 phi_sw 边界层厚度至 1.5-2.5 以增强对攻击跳变的鲁棒性）
        sat_s = np.tanh(s / self.phi_sw)
        dot_s_target = -self.K_s @ s - self.K_sw @ sat_s

        # 3. 计算等效控制需要的中间项 （鲁棒化处理：增大 epsilon 防止分母过小导致控制量因攻击数据剧震）
        epsilon = 0.05
        # coeff_beta = beta * b * |v_e|^(b-1)
        coeff_beta = self.beta_s @ np.diag(self.b_s * (np.abs(self.d_error)**(self.b_s - 1) + epsilon))
        # term_alpha = alpha * a * |e|^(a-1) * v_e
        term_alpha = self.alpha_s @ np.diag(self.a_s * (np.abs(self.p_error)**(self.a_s - 1))) @ self.d_error
        # term_lam = lam * e
        term_lam = self.lam_s @ self.p_error

        # 4. 解出 a_smc （公式推导：dot{s} = v_e + term_alpha + coeff_beta @ (a_smc - ad) + term_lam = dot_s_target）（整理得：a_smc = ad + inv(coeff_beta) @ (dot_s_target - v_error - term_alpha - term_lam)）
        inv_coeff_beta = np.linalg.inv(coeff_beta)
        # 鲁棒化处理：限制逆矩阵增益的最大值，防止受攻击时产生天文数字般的加速度
        inv_coeff_beta = np.clip(inv_coeff_beta, -5.0, 5.0)
        a_smc = ad + inv_coeff_beta @ (dot_s_target - self.d_error - term_alpha - term_lam)
        
        gravity = np.array([0., 0., -self.params['g']])
        F_r = self.params['m'] * (a_smc - gravity)
        
        try:
            lam = np.exp(- self.dt / self.params['force_filter_time_const'])
            self.F_r_dot *= lam
            self.F_r_dot += (1 - lam) * (F_r - self.F_r_last) * self.dt_inv
        except TypeError as err:
            if self.F_r_last is None:
                self.F_r_dot = np.zeros(3)
            else:
                raise err
        
        if any(np.isinf(self.F_r_dot)):
            raise ValueError
        self.F_r_last = F_r.copy()
        
        if logentry is not None:
            logentry['residual'] = residual
            logentry['p_purified'] = p_purified
            logentry['x_pred'] = self.x_pred
            logentry['e'] = self.e
            logentry['dot_e'] = self.dot_e
            logentry['int_e'] = self.int_e
            logentry['e_p'] = self.e_p
            logentry['dot_e_p'] = self.dot_e_p
            logentry['int_e_p'] = self.int_e_p
            logentry['p_error'] = self.p_error
            logentry['d_error'] = self.d_error
            logentry['i_error'] = self.i_error
            logentry['term_p'] = term_p
            logentry['term_v'] = term_d
            logentry['term_i'] = term_i
            logentry['s'] = s
            logentry['sat_s'] = sat_s
            logentry['dot_s_target'] = dot_s_target
            logentry['coeff_beta'] = coeff_beta
            logentry['term_alpha'] = term_alpha
            logentry['term_lam'] = term_lam
            logentry['inv_coeff_beta'] = inv_coeff_beta
            logentry['a_smc'] = a_smc
            logentry['ad_term'] = ad * self.params['m']
            logentry['jd_term'] = jd * self.params['m']
            logentry['g_term'] = np.array([0., 0., self.params['m'] * self.params['g']])

        return Force(F_r, self.F_r_dot)


class MetaAdapt(Baseline):

    _name = 'omac'
    name_long = 'OMAC'

    def __init__(self, integral_control=False, train=True, *args, **kwargs):
        super().__init__(integral_control=integral_control, *args, **kwargs)
        # Values to be initialized in reset_controller()
        self.motor_speed_command = None
        self.train = train

        # buffer holds recent tuples (X, fhat, y, a) for online adaptation
        self.replay_buffer = []
        self.replay_buffer_size = 200
        # online meta-update hyperparameters (used if model has `phi`)
        self.online_lr = 1e-3
        self.online_steps = 5
        self.online_batch_size = 16
        # residual magnitude threshold to trigger online update
        self.residual_threshold = 10  # if None, will not auto-trigger based on magnitude
        
        self.__local_rng = np.random.default_rng(52)

    def calculate_gains(self):
        super().calculate_gains()
        self.Lambda = np.linalg.solve(self.params['K_d'], self.params['K_p'])

    def reset_controller(self):
        ret = super().reset_controller()
        if ret is not None:
            raise ValueError
        self.motor_speed_command = np.zeros(4)
    
    def mixer(self, torque_sp, T_sp, logentry):
        ''' Override super().mixer() in order to save the output '''
        self.motor_speed_command = super().mixer(torque_sp, T_sp, logentry)
        return self.motor_speed_command

    def get_residual(self, X, imu, logentry):
        ''' Compute a measurement of the residual force '''
        q = X[3:7]
        R = rowan.to_matrix(q)  # body to world
        H = self.params['m'] * np.eye(3)
        g = np.array((0., 0., self.params['g'] * self.params['m']))
        T = self.params['C_T'] * sum(self.motor_speed_command ** 2)
        u = T * R @ np.array((0., 0., 1.))
        y = (H @ imu[0:3] + g - u)

        logentry['y'] = y
        logentry['g'] = g
        logentry['u'] = - u
        logentry['mpddot'] = H @ imu[0:3]

        return y

    def add_to_replay(self, X, fhat, y, a=None):
        """Add a sample to the replay buffer (keeps bounded size)."""
        # store numpy arrays (copy to avoid later mutation)
        item = (np.copy(X), np.copy(fhat), np.copy(y), None if a is None else np.copy(a))
        self.replay_buffer.append(item)
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

    def online_meta_update(self, wind=None):
        """
        Perform a small number of online meta-update steps using the replay buffer.
        This implementation only runs if the controller implements a `phi` network (e.g., MetaAdaptDeep / NeuralFly).
        It performs a few gradient steps on a small minibatch sampled from the replay buffer.
        """

        if not self.train:
            return

        if not hasattr(self, 'phi') or self.phi is None:
            return

        if len(self.replay_buffer) == 0:
            return

        # sample a contiguous minibatch to preserve temporal structure
        batch_size = min(self.online_batch_size, len(self.replay_buffer))
        start_max = max(len(self.replay_buffer) - batch_size, 0)

        X_list = []
        a_list = []
        F_list = []
        for start in range(start_max, -1, -1):  # prefer most recent window, fall back older ones
            X_list.clear()
            a_list.clear()
            F_list.clear()
            for X_i, fhat_i, y_i, a_i in self.replay_buffer[start:start + batch_size]:
                a_vec = None
                if a_i is not None:
                    a_vec = a_i
                elif hasattr(self, 'a') and self.a is not None:
                    a_vec = self.a
                if a_vec is None:
                    continue
                X_list.append(X_i)
                a_list.append(a_vec)
                F_list.append(y_i)
            if len(X_list) > 0:
                break

        if len(X_list) == 0:
            return

        # prepare device and dtype matching phi
        dev = next(self.phi.parameters()).device
        dtype = next(self.phi.parameters()).dtype
        # convert F targets and a vectors to tensors on correct device
        F_batch = torch.stack([torch.from_numpy(f).to(device=dev, dtype=dtype) for f in F_list])
        a_tensors = [torch.from_numpy(a).to(device=dev, dtype=dtype) for a in a_list]

        opt = optim.SGD(self.phi.parameters(), lr=self.online_lr)
        loss_fn = nn.MSELoss()
        for _ in range(self.online_steps):
            # recompute predictions each iteration to build fresh autograd graph
            if hasattr(self.phi, 'reset_hidden'):
                self.phi.reset_hidden()
            y_preds = []
            for X_i, a_t in zip(X_list, a_tensors):
                x_t = torch.from_numpy(X_i).to(device=dev, dtype=dtype)
                phi_out = self.phi(x_t)
                kron_mat = torch.kron(torch.eye(3, device=dev, dtype=dtype), phi_out)
                y_pred = torch.matmul(kron_mat, a_t)
                y_preds.append(y_pred)
            y_batch = torch.stack(y_preds)

            opt.zero_grad()
            loss = loss_fn(y_batch, F_batch)
            loss.backward()
            opt.step()
    
    def get_Fr(self, X, imu=np.zeros(3), pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), meta_adapt_trigger=None, wind=None, attack_active=None, logentry=None, **kwargs):
        ''' Override baseline force controller to also account for the adapted force '''
        # Get prior measurements. Note these can be the filtered measurements!
        y = self.get_residual(X, imu, logentry=logentry)
        fhat = self.get_f_hat(X)

        # Update parameters
        self.inner_adapt(X, fhat.F, y)
        self.update_batch(X, fhat.F, y)
        
        # store recent experience for continual/online meta-learning
        if attack_active and self.train and self.name_long in ['OMAC (deepRnnAttSafe)', 'OMAC (deepRnnAttSafeFly)', 'OMAC-PURE (deepRnnAttSafe)', 'OMAC-PURE (deepRnnAttSafeFly)', 'OMAC-SMC (deepRnnAttSafe)', 'OMAC-SMC (deepRnnAttSafeFly)', 'OMAC-PURE-SMC (deepRnnAttSafe)', 'OMAC-PURE-SMC (deepRnnAttSafeFly)']:
            self.add_to_replay(X, fhat.F, y, getattr(self, 'a', None))
        # if residual threshold is configured, trigger a short online update when residual is large
        if (self.residual_threshold is not None) and (np.linalg.norm(y) > self.residual_threshold) and self.train and self.name_long in ['OMAC (deepRnnAttSafe)', 'OMAC (deepRnnAttSafeFly)', 'OMAC-PURE (deepRnnAttSafe)', 'OMAC-PURE (deepRnnAttSafeFly)', 'OMAC-SMC (deepRnnAttSafe)', 'OMAC-SMC (deepRnnAttSafeFly)', 'OMAC-PURE-SMC (deepRnnAttSafe)', 'OMAC-PURE-SMC (deepRnnAttSafeFly)']:
            self.online_meta_update(wind=wind)
        
        if meta_adapt_trigger and self.train:
            self.meta_adapt(wind=wind)

        # Get baseline controller force (without integral control)
        Fr = super().get_Fr(X, imu, pd, vd, ad, jd, meta_adapt_trigger, wind, attack_active, logentry)
        F_r, F_r_dot, F_r_ddot = Fr

        # Add in adaptive term
        f_hat = self.get_f_hat(X)
        F_r -= f_hat.F

        logentry['f_hat'] = f_hat
        logentry['pred_error'] = y - f_hat
        logentry['F_r_no_adaptive'] = F_r.copy()

        return Force(F_r, F_r_dot, F_r_ddot)

    def get_f_hat(self, X):
        ''' Returns Force() named tuple with f_hat and, optionally, its derivatives '''
        raise NotImplementedError

    def inner_adapt(self, X, fhat, y):
        raise NotImplementedError

    def update_batch(self, X, fhat, y):
        raise NotImplementedError

    def meta_adapt(self, wind=None):
        raise NotImplementedError


class MetaAdaptPure(BaselinePure):

    _name = 'omac-smc'
    name_long = 'OMAC-SMC'

    def __init__(self, integral_control=False, train=True, *args, **kwargs):
        super().__init__(integral_control=integral_control, *args, **kwargs)
        # Values to be initialized in reset_controller()
        self.motor_speed_command = None
        self.train = train

        # buffer holds recent tuples (X, fhat, y, a) for online adaptation
        self.replay_buffer = []
        self.replay_buffer_size = 200
        # online meta-update hyperparameters (used if model has `phi`)
        self.online_lr = 1e-3
        self.online_steps = 5
        self.online_batch_size = 16
        # residual magnitude threshold to trigger online update
        self.residual_threshold = 10  # if None, will not auto-trigger based on magnitude
        
        self.__local_rng = np.random.default_rng(52)

    def calculate_gains(self):
        super().calculate_gains()
        self.Lambda = np.linalg.solve(self.params['K_d'], self.params['K_p'])

    def reset_controller(self):
        ret = super().reset_controller()
        if ret is not None:
            raise ValueError
        self.motor_speed_command = np.zeros(4)
    
    def mixer(self, torque_sp, T_sp, logentry):
        ''' Override super().mixer() in order to save the output '''
        self.motor_speed_command = super().mixer(torque_sp, T_sp, logentry)
        return self.motor_speed_command

    def get_residual(self, X, imu, logentry):
        ''' Compute a measurement of the residual force '''
        q = X[3:7]
        R = rowan.to_matrix(q)  # body to world
        H = self.params['m'] * np.eye(3)
        g = np.array((0., 0., self.params['g'] * self.params['m']))
        T = self.params['C_T'] * sum(self.motor_speed_command ** 2)
        u = T * R @ np.array((0., 0., 1.))
        y = (H @ imu[0:3] + g - u)

        logentry['y'] = y
        logentry['g'] = g
        logentry['u'] = - u
        logentry['mpddot'] = H @ imu[0:3]

        return y

    def add_to_replay(self, X, fhat, y, a=None):
        """Add a sample to the replay buffer (keeps bounded size)."""
        # store numpy arrays (copy to avoid later mutation)
        item = (np.copy(X), np.copy(fhat), np.copy(y), None if a is None else np.copy(a))
        self.replay_buffer.append(item)
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

    def online_meta_update(self, wind=None):
        """
        Perform a small number of online meta-update steps using the replay buffer.
        This implementation only runs if the controller implements a `phi` network (e.g., MetaAdaptDeep / NeuralFly).
        It performs a few gradient steps on a small minibatch sampled from the replay buffer.
        """

        if not self.train:
            return

        if not hasattr(self, 'phi') or self.phi is None:
            return

        if len(self.replay_buffer) == 0:
            return

        # sample a contiguous minibatch to preserve temporal structure
        batch_size = min(self.online_batch_size, len(self.replay_buffer))
        start_max = max(len(self.replay_buffer) - batch_size, 0)

        X_list = []
        a_list = []
        F_list = []
        for start in range(start_max, -1, -1):  # prefer most recent window, fall back older ones
            X_list.clear()
            a_list.clear()
            F_list.clear()
            for X_i, fhat_i, y_i, a_i in self.replay_buffer[start:start + batch_size]:
                a_vec = None
                if a_i is not None:
                    a_vec = a_i
                elif hasattr(self, 'a') and self.a is not None:
                    a_vec = self.a
                if a_vec is None:
                    continue
                X_list.append(X_i)
                a_list.append(a_vec)
                F_list.append(y_i)
            if len(X_list) > 0:
                break

        if len(X_list) == 0:
            return

        # prepare device and dtype matching phi
        dev = next(self.phi.parameters()).device
        dtype = next(self.phi.parameters()).dtype
        # convert F targets and a vectors to tensors on correct device
        F_batch = torch.stack([torch.from_numpy(f).to(device=dev, dtype=dtype) for f in F_list])
        a_tensors = [torch.from_numpy(a).to(device=dev, dtype=dtype) for a in a_list]

        opt = optim.SGD(self.phi.parameters(), lr=self.online_lr)
        loss_fn = nn.MSELoss()
        for _ in range(self.online_steps):
            # recompute predictions each iteration to build fresh autograd graph
            if hasattr(self.phi, 'reset_hidden'):
                self.phi.reset_hidden()
            y_preds = []
            for X_i, a_t in zip(X_list, a_tensors):
                x_t = torch.from_numpy(X_i).to(device=dev, dtype=dtype)
                phi_out = self.phi(x_t)
                kron_mat = torch.kron(torch.eye(3, device=dev, dtype=dtype), phi_out)
                y_pred = torch.matmul(kron_mat, a_t)
                y_preds.append(y_pred)
            y_batch = torch.stack(y_preds)

            opt.zero_grad()
            loss = loss_fn(y_batch, F_batch)
            loss.backward()
            opt.step()
    
    def get_Fr(self, X, imu=np.zeros(3), pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), meta_adapt_trigger=None, wind=None, attack_active=None, logentry=None, **kwargs):
        ''' Override baseline force controller to also account for the adapted force '''
        # Get prior measurements. Note these can be the filtered measurements!
        y = self.get_residual(X, imu, logentry=logentry)
        fhat = self.get_f_hat(X)

        # Update parameters
        self.inner_adapt(X, fhat.F, y)
        self.update_batch(X, fhat.F, y)
        
        # store recent experience for continual/online meta-learning
        if attack_active and self.train and self.name_long in ['OMAC (deepRnnAttSafe)', 'OMAC (deepRnnAttSafeFly)', 'OMAC-PURE (deepRnnAttSafe)', 'OMAC-PURE (deepRnnAttSafeFly)', 'OMAC-SMC (deepRnnAttSafe)', 'OMAC-SMC (deepRnnAttSafeFly)', 'OMAC-PURE-SMC (deepRnnAttSafe)', 'OMAC-PURE-SMC (deepRnnAttSafeFly)']:
            self.add_to_replay(X, fhat.F, y, getattr(self, 'a', None))
        # if residual threshold is configured, trigger a short online update when residual is large
        if (self.residual_threshold is not None) and (np.linalg.norm(y) > self.residual_threshold) and self.train and self.name_long in ['OMAC (deepRnnAttSafe)', 'OMAC (deepRnnAttSafeFly)', 'OMAC-PURE (deepRnnAttSafe)', 'OMAC-PURE (deepRnnAttSafeFly)', 'OMAC-SMC (deepRnnAttSafe)', 'OMAC-SMC (deepRnnAttSafeFly)', 'OMAC-PURE-SMC (deepRnnAttSafe)', 'OMAC-PURE-SMC (deepRnnAttSafeFly)']:
            self.online_meta_update(wind=wind)
        
        if meta_adapt_trigger and self.train:
            self.meta_adapt(wind=wind)

        # Get baseline controller force (without integral control)
        Fr = super().get_Fr(X, imu, pd, vd, ad, jd, meta_adapt_trigger, wind, attack_active, logentry)
        F_r, F_r_dot, F_r_ddot = Fr

        # Add in adaptive term
        f_hat = self.get_f_hat(X)
        F_r -= f_hat.F

        logentry['f_hat'] = f_hat
        logentry['pred_error'] = y - f_hat
        logentry['F_r_no_adaptive'] = F_r.copy()

        return Force(F_r, F_r_dot, F_r_ddot)

    def get_f_hat(self, X):
        ''' Returns Force() named tuple with f_hat and, optionally, its derivatives '''
        raise NotImplementedError

    def inner_adapt(self, X, fhat, y):
        raise NotImplementedError

    def update_batch(self, X, fhat, y):
        raise NotImplementedError

    def meta_adapt(self, wind=None):
        raise NotImplementedError


class MetaAdaptSmc(BaselineSmc):

    _name = 'omac-smc'
    name_long = 'OMAC-SMC'

    def __init__(self, integral_control=False, train=True, *args, **kwargs):
        super().__init__(integral_control=integral_control, *args, **kwargs)
        # Values to be initialized in reset_controller()
        self.motor_speed_command = None
        self.train = train

        # buffer holds recent tuples (X, fhat, y, a) for online adaptation
        self.replay_buffer = []
        self.replay_buffer_size = 200
        # online meta-update hyperparameters (used if model has `phi`)
        self.online_lr = 1e-3
        self.online_steps = 5
        self.online_batch_size = 16
        # residual magnitude threshold to trigger online update
        self.residual_threshold = 10  # if None, will not auto-trigger based on magnitude
        
        self.__local_rng = np.random.default_rng(52)

    def calculate_gains(self):
        super().calculate_gains()
        self.Lambda = np.linalg.solve(self.params['K_d'], self.params['K_p'])

    def reset_controller(self):
        ret = super().reset_controller()
        if ret is not None:
            raise ValueError
        self.motor_speed_command = np.zeros(4)
    
    def mixer(self, torque_sp, T_sp, logentry):
        ''' Override super().mixer() in order to save the output '''
        self.motor_speed_command = super().mixer(torque_sp, T_sp, logentry)
        return self.motor_speed_command

    def get_residual(self, X, imu, logentry):
        ''' Compute a measurement of the residual force '''
        q = X[3:7]
        R = rowan.to_matrix(q)  # body to world
        H = self.params['m'] * np.eye(3)
        g = np.array((0., 0., self.params['g'] * self.params['m']))
        T = self.params['C_T'] * sum(self.motor_speed_command ** 2)
        u = T * R @ np.array((0., 0., 1.))
        y = (H @ imu[0:3] + g - u)

        logentry['y'] = y
        logentry['g'] = g
        logentry['u'] = - u
        logentry['mpddot'] = H @ imu[0:3]

        return y

    def add_to_replay(self, X, fhat, y, a=None):
        """Add a sample to the replay buffer (keeps bounded size)."""
        # store numpy arrays (copy to avoid later mutation)
        item = (np.copy(X), np.copy(fhat), np.copy(y), None if a is None else np.copy(a))
        self.replay_buffer.append(item)
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

    def online_meta_update(self, wind=None):
        """
        Perform a small number of online meta-update steps using the replay buffer.
        This implementation only runs if the controller implements a `phi` network (e.g., MetaAdaptDeep / NeuralFly).
        It performs a few gradient steps on a small minibatch sampled from the replay buffer.
        """

        if not self.train:
            return

        if not hasattr(self, 'phi') or self.phi is None:
            return

        if len(self.replay_buffer) == 0:
            return

        # sample a contiguous minibatch to preserve temporal structure
        batch_size = min(self.online_batch_size, len(self.replay_buffer))
        start_max = max(len(self.replay_buffer) - batch_size, 0)

        X_list = []
        a_list = []
        F_list = []
        for start in range(start_max, -1, -1):  # prefer most recent window, fall back older ones
            X_list.clear()
            a_list.clear()
            F_list.clear()
            for X_i, fhat_i, y_i, a_i in self.replay_buffer[start:start + batch_size]:
                a_vec = None
                if a_i is not None:
                    a_vec = a_i
                elif hasattr(self, 'a') and self.a is not None:
                    a_vec = self.a
                if a_vec is None:
                    continue
                X_list.append(X_i)
                a_list.append(a_vec)
                F_list.append(y_i)
            if len(X_list) > 0:
                break

        if len(X_list) == 0:
            return

        # prepare device and dtype matching phi
        dev = next(self.phi.parameters()).device
        dtype = next(self.phi.parameters()).dtype
        # convert F targets and a vectors to tensors on correct device
        F_batch = torch.stack([torch.from_numpy(f).to(device=dev, dtype=dtype) for f in F_list])
        a_tensors = [torch.from_numpy(a).to(device=dev, dtype=dtype) for a in a_list]

        opt = optim.SGD(self.phi.parameters(), lr=self.online_lr)
        loss_fn = nn.MSELoss()
        for _ in range(self.online_steps):
            # recompute predictions each iteration to build fresh autograd graph
            if hasattr(self.phi, 'reset_hidden'):
                self.phi.reset_hidden()
            y_preds = []
            for X_i, a_t in zip(X_list, a_tensors):
                x_t = torch.from_numpy(X_i).to(device=dev, dtype=dtype)
                phi_out = self.phi(x_t)
                kron_mat = torch.kron(torch.eye(3, device=dev, dtype=dtype), phi_out)
                y_pred = torch.matmul(kron_mat, a_t)
                y_preds.append(y_pred)
            y_batch = torch.stack(y_preds)

            opt.zero_grad()
            loss = loss_fn(y_batch, F_batch)
            loss.backward()
            opt.step()
    
    def get_Fr(self, X, imu=np.zeros(3), pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), meta_adapt_trigger=None, wind=None, attack_active=None, logentry=None, **kwargs):
        ''' Override baseline force controller to also account for the adapted force '''
        # Get prior measurements. Note these can be the filtered measurements!
        y = self.get_residual(X, imu, logentry=logentry)
        fhat = self.get_f_hat(X)

        # Update parameters
        self.inner_adapt(X, fhat.F, y)
        self.update_batch(X, fhat.F, y)
        
        # store recent experience for continual/online meta-learning
        if attack_active and self.train and self.name_long in ['OMAC (deepRnnAttSafe)', 'OMAC (deepRnnAttSafeFly)', 'OMAC-PURE (deepRnnAttSafe)', 'OMAC-PURE (deepRnnAttSafeFly)', 'OMAC-SMC (deepRnnAttSafe)', 'OMAC-SMC (deepRnnAttSafeFly)', 'OMAC-PURE-SMC (deepRnnAttSafe)', 'OMAC-PURE-SMC (deepRnnAttSafeFly)']:
            self.add_to_replay(X, fhat.F, y, getattr(self, 'a', None))
        # if residual threshold is configured, trigger a short online update when residual is large
        if (self.residual_threshold is not None) and (np.linalg.norm(y) > self.residual_threshold) and self.train and self.name_long in ['OMAC (deepRnnAttSafe)', 'OMAC (deepRnnAttSafeFly)', 'OMAC-PURE (deepRnnAttSafe)', 'OMAC-PURE (deepRnnAttSafeFly)', 'OMAC-SMC (deepRnnAttSafe)', 'OMAC-SMC (deepRnnAttSafeFly)', 'OMAC-PURE-SMC (deepRnnAttSafe)', 'OMAC-PURE-SMC (deepRnnAttSafeFly)']:
            self.online_meta_update(wind=wind)
        
        if meta_adapt_trigger and self.train:
            self.meta_adapt(wind=wind)

        # Get baseline controller force (without integral control)
        Fr = super().get_Fr(X, imu, pd, vd, ad, jd, meta_adapt_trigger, wind, attack_active, logentry)
        F_r, F_r_dot, F_r_ddot = Fr

        # Add in adaptive term
        f_hat = self.get_f_hat(X)
        F_r -= f_hat.F

        logentry['f_hat'] = f_hat
        logentry['pred_error'] = y - f_hat
        logentry['F_r_no_adaptive'] = F_r.copy()

        return Force(F_r, F_r_dot, F_r_ddot)

    def get_f_hat(self, X):
        ''' Returns Force() named tuple with f_hat and, optionally, its derivatives '''
        raise NotImplementedError

    def inner_adapt(self, X, fhat, y):
        raise NotImplementedError

    def update_batch(self, X, fhat, y):
        raise NotImplementedError

    def meta_adapt(self, wind=None):
        raise NotImplementedError


class MetaAdaptPureSmc(BaselinePureSmc):

    _name = 'omac-pure-smc'
    name_long = 'OMAC-PURE-SMC'

    def __init__(self, integral_control=False, train=True, *args, **kwargs):
        super().__init__(integral_control=integral_control, *args, **kwargs)
        # Values to be initialized in reset_controller()
        self.motor_speed_command = None
        self.train = train

        # buffer holds recent tuples (X, fhat, y, a) for online adaptation
        self.replay_buffer = []
        self.replay_buffer_size = 200
        # online meta-update hyperparameters (used if model has `phi`)
        self.online_lr = 1e-3
        self.online_steps = 5
        self.online_batch_size = 16
        # residual magnitude threshold to trigger online update
        self.residual_threshold = 10  # if None, will not auto-trigger based on magnitude
        
        self.__local_rng = np.random.default_rng(52)

    def calculate_gains(self):
        super().calculate_gains()
        self.Lambda = np.linalg.solve(self.params['K_d'], self.params['K_p'])

    def reset_controller(self):
        ret = super().reset_controller()
        if ret is not None:
            raise ValueError
        self.motor_speed_command = np.zeros(4)
    
    def mixer(self, torque_sp, T_sp, logentry):
        ''' Override super().mixer() in order to save the output '''
        self.motor_speed_command = super().mixer(torque_sp, T_sp, logentry)
        return self.motor_speed_command

    def get_residual(self, X, imu, logentry):
        ''' Compute a measurement of the residual force '''
        q = X[3:7]
        R = rowan.to_matrix(q)  # body to world
        H = self.params['m'] * np.eye(3)
        g = np.array((0., 0., self.params['g'] * self.params['m']))
        T = self.params['C_T'] * sum(self.motor_speed_command ** 2)
        u = T * R @ np.array((0., 0., 1.))
        y = (H @ imu[0:3] + g - u)

        logentry['y'] = y
        logentry['g'] = g
        logentry['u'] = - u
        logentry['mpddot'] = H @ imu[0:3]

        return y

    def add_to_replay(self, X, fhat, y, a=None):
        """Add a sample to the replay buffer (keeps bounded size)."""
        # store numpy arrays (copy to avoid later mutation)
        item = (np.copy(X), np.copy(fhat), np.copy(y), None if a is None else np.copy(a))
        self.replay_buffer.append(item)
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)

    def online_meta_update(self, wind=None):
        """
        Perform a small number of online meta-update steps using the replay buffer.
        This implementation only runs if the controller implements a `phi` network (e.g., MetaAdaptDeep / NeuralFly).
        It performs a few gradient steps on a small minibatch sampled from the replay buffer.
        """

        if not self.train:
            return

        if not hasattr(self, 'phi') or self.phi is None:
            return

        if len(self.replay_buffer) == 0:
            return

        # sample a contiguous minibatch to preserve temporal structure
        batch_size = min(self.online_batch_size, len(self.replay_buffer))
        start_max = max(len(self.replay_buffer) - batch_size, 0)

        X_list = []
        a_list = []
        F_list = []
        for start in range(start_max, -1, -1):  # prefer most recent window, fall back older ones
            X_list.clear()
            a_list.clear()
            F_list.clear()
            for X_i, fhat_i, y_i, a_i in self.replay_buffer[start:start + batch_size]:
                a_vec = None
                if a_i is not None:
                    a_vec = a_i
                elif hasattr(self, 'a') and self.a is not None:
                    a_vec = self.a
                if a_vec is None:
                    continue
                X_list.append(X_i)
                a_list.append(a_vec)
                F_list.append(y_i)
            if len(X_list) > 0:
                break

        if len(X_list) == 0:
            return

        # prepare device and dtype matching phi
        dev = next(self.phi.parameters()).device
        dtype = next(self.phi.parameters()).dtype
        # convert F targets and a vectors to tensors on correct device
        F_batch = torch.stack([torch.from_numpy(f).to(device=dev, dtype=dtype) for f in F_list])
        a_tensors = [torch.from_numpy(a).to(device=dev, dtype=dtype) for a in a_list]

        opt = optim.SGD(self.phi.parameters(), lr=self.online_lr)
        loss_fn = nn.MSELoss()
        for _ in range(self.online_steps):
            # recompute predictions each iteration to build fresh autograd graph
            if hasattr(self.phi, 'reset_hidden'):
                self.phi.reset_hidden()
            y_preds = []
            for X_i, a_t in zip(X_list, a_tensors):
                x_t = torch.from_numpy(X_i).to(device=dev, dtype=dtype)
                phi_out = self.phi(x_t)
                kron_mat = torch.kron(torch.eye(3, device=dev, dtype=dtype), phi_out)
                y_pred = torch.matmul(kron_mat, a_t)
                y_preds.append(y_pred)
            y_batch = torch.stack(y_preds)

            opt.zero_grad()
            loss = loss_fn(y_batch, F_batch)
            loss.backward()
            opt.step()
    
    def get_Fr(self, X, imu=np.zeros(3), pd=np.zeros(3), vd=np.zeros(3), ad=np.zeros(3), jd=np.zeros(3), meta_adapt_trigger=None, wind=None, attack_active=None, logentry=None, **kwargs):
        ''' Override baseline force controller to also account for the adapted force '''
        # Get prior measurements. Note these can be the filtered measurements!
        y = self.get_residual(X, imu, logentry=logentry)
        fhat = self.get_f_hat(X)

        # Update parameters
        self.inner_adapt(X, fhat.F, y)
        self.update_batch(X, fhat.F, y)
        
        # store recent experience for continual/online meta-learning
        if attack_active and self.train and self.name_long in ['OMAC (deepRnnAttSafe)', 'OMAC (deepRnnAttSafeFly)', 'OMAC-PURE (deepRnnAttSafe)', 'OMAC-PURE (deepRnnAttSafeFly)', 'OMAC-SMC (deepRnnAttSafe)', 'OMAC-SMC (deepRnnAttSafeFly)', 'OMAC-PURE-SMC (deepRnnAttSafe)', 'OMAC-PURE-SMC (deepRnnAttSafeFly)']:
            self.add_to_replay(X, fhat.F, y, getattr(self, 'a', None))
        # if residual threshold is configured, trigger a short online update when residual is large
        if (self.residual_threshold is not None) and (np.linalg.norm(y) > self.residual_threshold) and self.train and self.name_long in ['OMAC (deepRnnAttSafe)', 'OMAC (deepRnnAttSafeFly)', 'OMAC-PURE (deepRnnAttSafe)', 'OMAC-PURE (deepRnnAttSafeFly)', 'OMAC-SMC (deepRnnAttSafe)', 'OMAC-SMC (deepRnnAttSafeFly)', 'OMAC-PURE-SMC (deepRnnAttSafe)', 'OMAC-PURE-SMC (deepRnnAttSafeFly)']:
            self.online_meta_update(wind=wind)
        
        if meta_adapt_trigger and self.train:
            self.meta_adapt(wind=wind)

        # Get baseline controller force (without integral control)
        Fr = super().get_Fr(X, imu, pd, vd, ad, jd, meta_adapt_trigger, wind, attack_active, logentry)
        F_r, F_r_dot, F_r_ddot = Fr

        # Add in adaptive term
        f_hat = self.get_f_hat(X)
        F_r -= f_hat.F

        logentry['f_hat'] = f_hat
        logentry['pred_error'] = y - f_hat
        logentry['F_r_no_adaptive'] = F_r.copy()

        return Force(F_r, F_r_dot, F_r_ddot)

    def get_f_hat(self, X):
        ''' Returns Force() named tuple with f_hat and, optionally, its derivatives '''
        raise NotImplementedError

    def inner_adapt(self, X, fhat, y):
        raise NotImplementedError

    def update_batch(self, X, fhat, y):
        raise NotImplementedError

    def meta_adapt(self, wind=None):
        raise NotImplementedError


class MetaAdaptDeep(MetaAdapt):

    _name = 'omac-deep'
    name_long = 'OMAC (deep)'

    class Phi(nn.Module):
        def __init__(self, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(13, layer_sizes[0]))
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def reset_hidden(self):
            pass

        def feature(self, x_np):
            with torch.no_grad():
                phi_out = self(torch.from_numpy(x_np)).numpy()
            return np.kron(np.eye(3), phi_out)

    def __init__(self, dim_a=100, eta_a_base=0.010, eta_A_base=0.005, layer_sizes=(25, 50), train=True, *args, **kwargs):
        super().__init__(train=train, *args, **kwargs)
        # Initialize kernels
        self.layer_sizes = layer_sizes
        self.dim_a = dim_a - dim_a % 3

        # Initialize parameters
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base

        # Initialize learned parameter matrices
        self.a = np.zeros(self.dim_a)
        self.phi = self.Phi(dim_kernel=int(self.dim_a / 3), layer_sizes=self.layer_sizes)
        self.optimizer = optim.Adam(self.phi.parameters(), lr=self.eta_A_base)
        self.loss = nn.MSELoss()

    def reset_controller(self):
        super().reset_controller()

        self.inner_adapt_count = 0
        self.meta_adapt_count = 0

        self.reset_batch()

        self.a = np.zeros(self.dim_a)

        if hasattr(self, 'phi'):
            self.phi.reset_hidden()

    def get_phi(self, X):
        return self.phi.feature(X)

    def get_f_hat(self, X):
        phi = self.get_phi(X)
        return Force(phi @ self.a)

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = max(self.eta_a_base / np.sqrt(self.inner_adapt_count), self.eta_a_base * 0.2)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_phi(X)
        if (np.linalg.norm(self.a)>20):
            self.a = self.a / np.linalg.norm(self.a) * 20
    
    def update_batch(self, X, fhat, y):
        self.batch.append((X, y, self.a.copy()))

    def reset_batch(self):
        self.batch = []

    def meta_adapt(self, wind=None):
        if len(self.batch) < 50:
            return
        self.inner_adapt_count = 0
        self.meta_adapt_count += 1
        eta_A = self.eta_A_base / np.sqrt(self.meta_adapt_count)

        y_batch = []
        F_batch = []
        self.phi.reset_hidden()
        for X, y, a in self.batch[:-1]:
            x_t = torch.from_numpy(X)
            phi_out = self.phi(x_t)
            kron_mat = torch.kron(torch.eye(3), phi_out)
            y_pred = torch.matmul(kron_mat, torch.from_numpy(a))
            y_batch.append(y_pred)
            F_batch.append(torch.from_numpy(y))
        y_batch = torch.stack(y_batch)
        F_batch = torch.stack(F_batch)

        self.optimizer.zero_grad()
        loss = self.loss(y_batch, F_batch)
        loss.backward()
        self.optimizer.step()

        self.reset_batch()
        self.meta_adapt_count = 0


class MetaAdaptOod(MetaAdaptDeep):

    _name = 'omac-ood'
    name_long = 'OMAC (ood)'

    def __init__(self, dim_a=100, eta_a_base=0.010, eta_A_base=0.005, layer_sizes=(25, 50), noise_x=0.06, noise_a=0.06, train=True, *args, **kwargs):
        super().__init__(dim_a=dim_a, eta_a_base=eta_a_base, eta_A_base=eta_A_base, layer_sizes=layer_sizes, train=train, *args, **kwargs)
        self.noise_x = noise_x
        self.noise_a = noise_a
        self._local_rng_Ood = np.random.default_rng(51)
    
    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = max(self.eta_a_base / np.sqrt(self.inner_adapt_count), self.eta_a_base * 0.2)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_phi(X)
        if (np.linalg.norm(self.a)>20):
            self.a = self.a / np.linalg.norm(self.a) * 20

    def meta_adapt(self, wind=None):
        if len(self.batch) < 50:
            return
        self.inner_adapt_count = 0
        self.meta_adapt_count += 1
        eta_A = self.eta_A_base / np.sqrt(self.meta_adapt_count)

        y_batch = []
        F_batch = []
        self.phi.reset_hidden()
        for X, y, a in self.batch[:-1]:
            X = X + self.noise_x * self._local_rng_Ood.normal(0, 1, X.shape)
            a = a + self.noise_a * self._local_rng_Ood.normal(0, 1, a.shape)
            x_t = torch.from_numpy(X)
            phi_out = self.phi(x_t)
            kron_mat = torch.kron(torch.eye(3), phi_out)
            y_pred = torch.matmul(kron_mat, torch.from_numpy(a))
            y_batch.append(y_pred)
            F_batch.append(torch.from_numpy(y))
        y_batch = torch.stack(y_batch)
        F_batch = torch.stack(F_batch)

        self.optimizer.zero_grad()
        loss = self.loss(y_batch, F_batch)
        loss.backward()
        self.optimizer.step()

        self.reset_batch()
        self.meta_adapt_count = 0


class MetaAdaptNeuralFly(MetaAdaptDeep):

    _name = 'omac-neuralFly'
    name_long = '(OMAC) NeuralFly'

    class H(nn.Module):
        def __init__(self, start_kernel, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(start_kernel, layer_sizes[0]))
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    def __init__(self, dim_a=100, eta_a_base=0.010, eta_A_base=0.005, layer_sizes=(25, 50), train=True, *args, **kwargs):
        super().__init__(dim_a=dim_a, eta_a_base=eta_a_base, eta_A_base=eta_A_base, layer_sizes=layer_sizes, train=train, *args, **kwargs)
        self.h = self.H(start_kernel=int(self.dim_a/3), dim_kernel=16, layer_sizes=self.layer_sizes)
        self.h_optimizer = optim.Adam(params=self.h.parameters(), lr=0.005)
        self.h_loss = nn.CrossEntropyLoss()
        self.wind_idx = None
        self.beta = 0.5
        self.frequency_H = 2
        self.spectral_normalization = 2.0
        self._local_rng_NeuralFly = np.random.default_rng(51)
    
    def meta_adapt(self, wind=None):
        if len(self.batch) < 50:
            return
        self.inner_adapt_count = 0
        self.meta_adapt_count += 1
        eta_A = self.eta_A_base / np.sqrt(self.meta_adapt_count)
        
        if wind is not None:
            self.wind_idx = np.floor(np.linalg.norm(wind))
        if self.wind_idx > 15.0:
            self.wind_idx = 15.0
        if self.wind_idx < 0.0:
            self.wind_idx = 0.0
        wind_tgt = torch.tensor([int(self.wind_idx) for _ in range(len(self.batch[:-1]))], dtype=int)
        y_batch = []
        F_batch = []
        H_batch = []
        self.phi.reset_hidden()
        for X, y, a in self.batch[:-1]:
            x_t = torch.from_numpy(X)
            phi_out = self.phi(x_t)
            kron_mat = torch.kron(torch.eye(3), phi_out)
            y_pred = torch.matmul(kron_mat, torch.from_numpy(a))
            h_logit = self.h(phi_out)
            y_batch.append(y_pred)
            F_batch.append(torch.from_numpy(y))
            H_batch.append(h_logit)
        y_batch = torch.stack(y_batch)
        F_batch = torch.stack(F_batch)
        H_batch = torch.stack(H_batch)
        self.optimizer.zero_grad()
        loss = self.loss(y_batch, F_batch) + self.beta * self.h_loss(H_batch, wind_tgt)
        loss.backward()
        self.optimizer.step()

        if (self._local_rng_NeuralFly.random() < 1 / self.frequency_H):
            if wind is not None:
                self.wind_idx = np.floor(np.linalg.norm(wind))
            if self.wind_idx > 15.0:
                self.wind_idx = 15.0
            if self.wind_idx < 0.0:
                self.wind_idx = 0.0
            wind_tgt = torch.tensor([int(self.wind_idx) for _ in range(len(self.batch))], dtype=int)
            H_batch = []
            self.phi.reset_hidden()
            for X, y, a in self.batch:
                x_t = torch.from_numpy(X)
                phi_out = self.phi(x_t).detach()
                h_logit = self.h(phi_out)
                H_batch.append(h_logit)
            H_batch = torch.stack(H_batch)
            self.h_optimizer.zero_grad()
            loss_h = self.h_loss(H_batch, wind_tgt)
            loss_h.backward()
            self.h_optimizer.step()
        
        # Spectral normalization
        if self.spectral_normalization > 0:
            for param in self.phi.parameters():
                M = param.detach().cpu().numpy()
                if M.ndim > 1:
                    s = np.linalg.norm(M, 2)
                    if s > self.spectral_normalization:
                        param.data = param / s * self.spectral_normalization

        self.reset_batch()
        self.meta_adapt_count = 0


class LuongAttention(nn.Module):
    """Luong dot-product attention for cross-RNN-layer feature aggregation.
    Projects query to key space for scoring, projects context back to query space for residual add.
    """
    def __init__(self, query_dim, key_dim):
        super().__init__()
        self.proj_in = spectral_norm(nn.Linear(query_dim, key_dim))
        self.proj_out = spectral_norm(nn.Linear(key_dim, query_dim))

    def forward(self, query, keys, values):
        # query: (seq_q, batch, query_dim)
        # keys:  (seq_k, batch, key_dim)
        # values:(seq_v, batch, key_dim)
        q = query.transpose(0, 1)   # (B, seq_q, query_dim)
        k = keys.transpose(0, 1)    # (B, seq_k, key_dim)
        v = values.transpose(0, 1)  # (B, seq_v, key_dim)

        q_proj = self.proj_in(q)                         # (B, seq_q, key_dim)
        scores = torch.bmm(q_proj, k.transpose(1, 2))    # (B, seq_q, seq_k)
        attn   = F.softmax(scores, dim=-1)               # (B, seq_q, seq_k)
        ctx    = torch.bmm(attn, v)                      # (B, seq_q, key_dim)
        ctx    = self.proj_out(ctx)                      # (B, seq_q, query_dim)
        return ctx.squeeze(1)                            # (B, query_dim)


class MetaAdaptDeepRnnAtt(MetaAdapt):

    _name = 'omac-deepRnnAtt'
    name_long = 'OMAC (deepRnnAtt)'

    class Phi(nn.Module):
        def __init__(self, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(13, layer_sizes[0]))
            self.rnn1 = nn.RNN(layer_sizes[0], layer_sizes[0], 2)
            self.rnn1 = self._sn_rnn(self.rnn1)
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.rnn2 = nn.RNN(layer_sizes[1], layer_sizes[1], 2)
            self.rnn2 = self._sn_rnn(self.rnn2)
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))
            self.att = LuongAttention(query_dim=layer_sizes[1], key_dim=layer_sizes[0])
            self._h1 = None
            self._h2 = None

        @staticmethod
        def _sn_rnn(rnn_layer):
            for name, _ in list(rnn_layer.named_parameters()):
                if 'weight' in name:
                    spectral_norm(rnn_layer, name)
            return rnn_layer

        def reset_hidden(self):
            self._h1 = None
            self._h2 = None

        def forward(self, x, detach_hidden=True):
            squeeze_out = x.dim() == 1
            if squeeze_out:
                x = x.unsqueeze(0)
            if x.dim() == 2:
                x = x.unsqueeze(0)

            T, B, _ = x.shape
            x_fc1 = F.relu(self.fc1(x.reshape(T * B, -1))).reshape(T, B, -1)
            out1, h1 = self.rnn1(x_fc1, self._h1)
            self._h1 = h1.detach() if detach_hidden else h1

            x_fc2 = F.relu(self.fc2(out1.reshape(T * B, -1))).reshape(T, B, -1)
            out2, h2 = self.rnn2(x_fc2, self._h2)
            self._h2 = h2.detach() if detach_hidden else h2

            ctx = self.att(out2[-1:], self._h1, self._h1)
            feat = out2[-1] + ctx
            out = self.fc3(feat)

            if squeeze_out:
                out = out.squeeze(0)
            return out

        def feature(self, x_np):
            with torch.no_grad():
                phi_out = self(torch.from_numpy(x_np)).numpy()
            return np.kron(np.eye(3), phi_out)

    def __init__(self, dim_a=100, eta_a_base=0.010, eta_A_base=0.005, layer_sizes=(25, 50), train=True, *args, **kwargs):
        super().__init__(train=train, *args, **kwargs)
        # Initialize kernels
        self.layer_sizes = layer_sizes
        self.dim_a = dim_a - dim_a % 3
        
        # Initialize parameters
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base

        # Initialize learned parameter matrices
        self.a = np.zeros(self.dim_a)
        self.phi = self.Phi(dim_kernel=int(self.dim_a / 3), layer_sizes=self.layer_sizes)
        self.optimizer = optim.Adam(self.phi.parameters(), lr=self.eta_A_base)
        self.loss = nn.MSELoss()
    
    def reset_controller(self):
        super().reset_controller()

        self.inner_adapt_count = 0
        self.meta_adapt_count = 0

        self.reset_batch()

        self.a = np.zeros(self.dim_a)

        if hasattr(self, 'phi'):
            self.phi.reset_hidden()

    def get_phi(self, X):
        return self.phi.feature(X)

    def get_f_hat(self, X):
        phi = self.get_phi(X)
        return Force(phi @ self.a)

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = max(self.eta_a_base / np.sqrt(self.inner_adapt_count), self.eta_a_base * 0.2)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_phi(X)
        if (np.linalg.norm(self.a)>20):
            self.a = self.a / np.linalg.norm(self.a) * 20

    def update_batch(self, X, fhat, y):
        self.batch.append((X, y, self.a.copy()))

    def reset_batch(self):
        self.batch = []

    def meta_adapt(self, wind=None):
        if len(self.batch) < 50:
            return
        self.inner_adapt_count = 0
        self.meta_adapt_count += 1
        eta_A = self.eta_A_base / np.sqrt(self.meta_adapt_count)

        y_batch = []
        F_batch = []
        self.phi.reset_hidden()
        for X, y, a in self.batch[:-1]:
            x_t = torch.from_numpy(X)
            phi_out = self.phi(x_t, detach_hidden=False)
            kron_mat = torch.kron(torch.eye(3), phi_out)
            y_pred = torch.matmul(kron_mat, torch.from_numpy(a))
            y_batch.append(y_pred)
            F_batch.append(torch.from_numpy(y))
        y_batch = torch.stack(y_batch)
        F_batch = torch.stack(F_batch)

        self.optimizer.zero_grad()
        loss = self.loss(y_batch, F_batch)
        loss.backward()
        self.optimizer.step()

        self.reset_batch()
        self.meta_adapt_count = 0


class MetaAdaptPureDeep(MetaAdaptPure):

    _name = 'omac-pure-deep'
    name_long = 'OMAC-PURE (deep)'

    class Phi(nn.Module):
        def __init__(self, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(13, layer_sizes[0]))
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def reset_hidden(self):
            pass

        def feature(self, x_np):
            with torch.no_grad():
                phi_out = self(torch.from_numpy(x_np)).numpy()
            return np.kron(np.eye(3), phi_out)

    def __init__(self, dim_a=100, eta_a_base=0.010, eta_A_base=0.005, layer_sizes=(25, 50), train=True, *args, **kwargs):
        super().__init__(train=train, *args, **kwargs)
        # Initialize kernels
        self.layer_sizes = layer_sizes
        self.dim_a = dim_a - dim_a % 3

        # Initialize parameters
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base

        # Initialize learned parameter matrices
        self.a = np.zeros(self.dim_a)
        self.phi = self.Phi(dim_kernel=int(self.dim_a / 3), layer_sizes=self.layer_sizes)
        self.optimizer = optim.Adam(self.phi.parameters(), lr=self.eta_A_base)
        self.loss = nn.MSELoss()

    def reset_controller(self):
        super().reset_controller()

        self.inner_adapt_count = 0
        self.meta_adapt_count = 0

        self.reset_batch()

        self.a = np.zeros(self.dim_a)

        if hasattr(self, 'phi'):
            self.phi.reset_hidden()

    def get_phi(self, X):
        return self.phi.feature(X)

    def get_f_hat(self, X):
        phi = self.get_phi(X)
        return Force(phi @ self.a)

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = max(self.eta_a_base / np.sqrt(self.inner_adapt_count), self.eta_a_base * 0.2)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_phi(X)
        if (np.linalg.norm(self.a)>20):
            self.a = self.a / np.linalg.norm(self.a) * 20
    
    def update_batch(self, X, fhat, y):
        self.batch.append((X, y, self.a.copy()))

    def reset_batch(self):
        self.batch = []

    def meta_adapt(self, wind=None):
        if len(self.batch) < 50:
            return
        self.inner_adapt_count = 0
        self.meta_adapt_count += 1
        eta_A = self.eta_A_base / np.sqrt(self.meta_adapt_count)

        y_batch = []
        F_batch = []
        self.phi.reset_hidden()
        for X, y, a in self.batch[:-1]:
            x_t = torch.from_numpy(X)
            phi_out = self.phi(x_t)
            kron_mat = torch.kron(torch.eye(3), phi_out)
            y_pred = torch.matmul(kron_mat, torch.from_numpy(a))
            y_batch.append(y_pred)
            F_batch.append(torch.from_numpy(y))
        y_batch = torch.stack(y_batch)
        F_batch = torch.stack(F_batch)

        self.optimizer.zero_grad()
        loss = self.loss(y_batch, F_batch)
        loss.backward()
        self.optimizer.step()

        self.reset_batch()
        self.meta_adapt_count = 0


class MetaAdaptPureDeepRnnAtt(MetaAdaptPure):

    _name = 'omac-pure-deepRnnAtt'
    name_long = 'OMAC-PURE (deepRnnAtt)'

    class Phi(nn.Module):
        def __init__(self, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(13, layer_sizes[0]))
            self.rnn1 = nn.RNN(layer_sizes[0], layer_sizes[0], 2)
            self.rnn1 = self._sn_rnn(self.rnn1)
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.rnn2 = nn.RNN(layer_sizes[1], layer_sizes[1], 2)
            self.rnn2 = self._sn_rnn(self.rnn2)
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))
            self.att = LuongAttention(query_dim=layer_sizes[1], key_dim=layer_sizes[0])
            self._h1 = None
            self._h2 = None

        @staticmethod
        def _sn_rnn(rnn_layer):
            for name, _ in list(rnn_layer.named_parameters()):
                if 'weight' in name:
                    spectral_norm(rnn_layer, name)
            return rnn_layer

        def reset_hidden(self):
            self._h1 = None
            self._h2 = None

        def forward(self, x, detach_hidden=True):
            squeeze_out = x.dim() == 1
            if squeeze_out:
                x = x.unsqueeze(0)
            if x.dim() == 2:
                x = x.unsqueeze(0)

            T, B, _ = x.shape
            x_fc1 = F.relu(self.fc1(x.reshape(T * B, -1))).reshape(T, B, -1)
            out1, h1 = self.rnn1(x_fc1, self._h1)
            self._h1 = h1.detach() if detach_hidden else h1

            x_fc2 = F.relu(self.fc2(out1.reshape(T * B, -1))).reshape(T, B, -1)
            out2, h2 = self.rnn2(x_fc2, self._h2)
            self._h2 = h2.detach() if detach_hidden else h2

            ctx = self.att(out2[-1:], self._h1, self._h1)
            feat = out2[-1] + ctx
            out = self.fc3(feat)

            if squeeze_out:
                out = out.squeeze(0)
            return out

        def feature(self, x_np):
            with torch.no_grad():
                phi_out = self(torch.from_numpy(x_np)).numpy()
            return np.kron(np.eye(3), phi_out)

    def __init__(self, dim_a=100, eta_a_base=0.010, eta_A_base=0.005, layer_sizes=(25, 50), train=True, *args, **kwargs):
        super().__init__(train=train, *args, **kwargs)
        # Initialize kernels
        self.layer_sizes = layer_sizes
        self.dim_a = dim_a - dim_a % 3
        
        # Initialize parameters
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base

        # Initialize learned parameter matrices
        self.a = np.zeros(self.dim_a)
        self.phi = self.Phi(dim_kernel=int(self.dim_a / 3), layer_sizes=self.layer_sizes)
        self.optimizer = optim.Adam(self.phi.parameters(), lr=self.eta_A_base)
        self.loss = nn.MSELoss()
    
    def reset_controller(self):
        super().reset_controller()

        self.inner_adapt_count = 0
        self.meta_adapt_count = 0

        self.reset_batch()

        self.a = np.zeros(self.dim_a)

        if hasattr(self, 'phi'):
            self.phi.reset_hidden()

    def get_phi(self, X):
        return self.phi.feature(X)

    def get_f_hat(self, X):
        phi = self.get_phi(X)
        return Force(phi @ self.a)

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = max(self.eta_a_base / np.sqrt(self.inner_adapt_count), self.eta_a_base * 0.2)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_phi(X)
        if (np.linalg.norm(self.a)>20):
            self.a = self.a / np.linalg.norm(self.a) * 20

    def update_batch(self, X, fhat, y):
        self.batch.append((X, y, self.a.copy()))

    def reset_batch(self):
        self.batch = []

    def meta_adapt(self, wind=None):
        if len(self.batch) < 50:
            return
        self.inner_adapt_count = 0
        self.meta_adapt_count += 1
        eta_A = self.eta_A_base / np.sqrt(self.meta_adapt_count)

        y_batch = []
        F_batch = []
        self.phi.reset_hidden()
        for X, y, a in self.batch[:-1]:
            x_t = torch.from_numpy(X)
            phi_out = self.phi(x_t, detach_hidden=False)
            kron_mat = torch.kron(torch.eye(3), phi_out)
            y_pred = torch.matmul(kron_mat, torch.from_numpy(a))
            y_batch.append(y_pred)
            F_batch.append(torch.from_numpy(y))
        y_batch = torch.stack(y_batch)
        F_batch = torch.stack(F_batch)

        self.optimizer.zero_grad()
        loss = self.loss(y_batch, F_batch)
        loss.backward()
        self.optimizer.step()

        self.reset_batch()
        self.meta_adapt_count = 0


class MetaAdaptSmcDeep(MetaAdaptSmc):

    _name = 'omac-smc-deep'
    name_long = 'OMAC-SMC (deep)'

    class Phi(nn.Module):
        def __init__(self, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(13, layer_sizes[0]))
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def reset_hidden(self):
            pass

        def feature(self, x_np):
            with torch.no_grad():
                phi_out = self(torch.from_numpy(x_np)).numpy()
            return np.kron(np.eye(3), phi_out)

    def __init__(self, dim_a=100, eta_a_base=0.010, eta_A_base=0.005, layer_sizes=(25, 50), train=True, *args, **kwargs):
        super().__init__(train=train, *args, **kwargs)
        # Initialize kernels
        self.layer_sizes = layer_sizes
        self.dim_a = dim_a - dim_a % 3

        # Initialize parameters
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base

        # Initialize learned parameter matrices
        self.a = np.zeros(self.dim_a)
        self.phi = self.Phi(dim_kernel=int(self.dim_a / 3), layer_sizes=self.layer_sizes)
        self.optimizer = optim.Adam(self.phi.parameters(), lr=self.eta_A_base)
        self.loss = nn.MSELoss()

    def reset_controller(self):
        super().reset_controller()

        self.inner_adapt_count = 0
        self.meta_adapt_count = 0

        self.reset_batch()

        self.a = np.zeros(self.dim_a)

        if hasattr(self, 'phi'):
            self.phi.reset_hidden()

    def get_phi(self, X):
        return self.phi.feature(X)

    def get_f_hat(self, X):
        phi = self.get_phi(X)
        return Force(phi @ self.a)

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = max(self.eta_a_base / np.sqrt(self.inner_adapt_count), self.eta_a_base * 0.2)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_phi(X)
        if (np.linalg.norm(self.a)>20):
            self.a = self.a / np.linalg.norm(self.a) * 20
    
    def update_batch(self, X, fhat, y):
        self.batch.append((X, y, self.a.copy()))

    def reset_batch(self):
        self.batch = []

    def meta_adapt(self, wind=None):
        if len(self.batch) < 50:
            return
        self.inner_adapt_count = 0
        self.meta_adapt_count += 1
        eta_A = self.eta_A_base / np.sqrt(self.meta_adapt_count)

        y_batch = []
        F_batch = []
        self.phi.reset_hidden()
        for X, y, a in self.batch[:-1]:
            x_t = torch.from_numpy(X)
            phi_out = self.phi(x_t)
            kron_mat = torch.kron(torch.eye(3), phi_out)
            y_pred = torch.matmul(kron_mat, torch.from_numpy(a))
            y_batch.append(y_pred)
            F_batch.append(torch.from_numpy(y))
        y_batch = torch.stack(y_batch)
        F_batch = torch.stack(F_batch)

        self.optimizer.zero_grad()
        loss = self.loss(y_batch, F_batch)
        loss.backward()
        self.optimizer.step()

        self.reset_batch()
        self.meta_adapt_count = 0


class MetaAdaptSmcDeepRnnAtt(MetaAdaptSmc):

    _name = 'omac-smc-deepRnnAtt'
    name_long = 'OMAC-SMC (deepRnnAtt)'

    class Phi(nn.Module):
        def __init__(self, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(13, layer_sizes[0]))
            self.rnn1 = nn.RNN(layer_sizes[0], layer_sizes[0], 2)
            self.rnn1 = self._sn_rnn(self.rnn1)
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.rnn2 = nn.RNN(layer_sizes[1], layer_sizes[1], 2)
            self.rnn2 = self._sn_rnn(self.rnn2)
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))
            self.att = LuongAttention(query_dim=layer_sizes[1], key_dim=layer_sizes[0])
            self._h1 = None
            self._h2 = None

        @staticmethod
        def _sn_rnn(rnn_layer):
            for name, _ in list(rnn_layer.named_parameters()):
                if 'weight' in name:
                    spectral_norm(rnn_layer, name)
            return rnn_layer

        def reset_hidden(self):
            self._h1 = None
            self._h2 = None

        def forward(self, x, detach_hidden=True):
            squeeze_out = x.dim() == 1
            if squeeze_out:
                x = x.unsqueeze(0)
            if x.dim() == 2:
                x = x.unsqueeze(0)

            T, B, _ = x.shape
            x_fc1 = F.relu(self.fc1(x.reshape(T * B, -1))).reshape(T, B, -1)
            out1, h1 = self.rnn1(x_fc1, self._h1)
            self._h1 = h1.detach() if detach_hidden else h1

            x_fc2 = F.relu(self.fc2(out1.reshape(T * B, -1))).reshape(T, B, -1)
            out2, h2 = self.rnn2(x_fc2, self._h2)
            self._h2 = h2.detach() if detach_hidden else h2

            ctx = self.att(out2[-1:], self._h1, self._h1)
            feat = out2[-1] + ctx
            out = self.fc3(feat)

            if squeeze_out:
                out = out.squeeze(0)
            return out

        def feature(self, x_np):
            with torch.no_grad():
                phi_out = self(torch.from_numpy(x_np)).numpy()
            return np.kron(np.eye(3), phi_out)

    def __init__(self, dim_a=100, eta_a_base=0.010, eta_A_base=0.005, layer_sizes=(25, 50), train=True, *args, **kwargs):
        super().__init__(train=train, *args, **kwargs)
        # Initialize kernels
        self.layer_sizes = layer_sizes
        self.dim_a = dim_a - dim_a % 3
        
        # Initialize parameters
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base

        # Initialize learned parameter matrices
        self.a = np.zeros(self.dim_a)
        self.phi = self.Phi(dim_kernel=int(self.dim_a / 3), layer_sizes=self.layer_sizes)
        self.optimizer = optim.Adam(self.phi.parameters(), lr=self.eta_A_base)
        self.loss = nn.MSELoss()
    
    def reset_controller(self):
        super().reset_controller()

        self.inner_adapt_count = 0
        self.meta_adapt_count = 0

        self.reset_batch()

        self.a = np.zeros(self.dim_a)

        if hasattr(self, 'phi'):
            self.phi.reset_hidden()

    def get_phi(self, X):
        return self.phi.feature(X)

    def get_f_hat(self, X):
        phi = self.get_phi(X)
        return Force(phi @ self.a)

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = max(self.eta_a_base / np.sqrt(self.inner_adapt_count), self.eta_a_base * 0.2)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_phi(X)
        if (np.linalg.norm(self.a)>20):
            self.a = self.a / np.linalg.norm(self.a) * 20

    def update_batch(self, X, fhat, y):
        self.batch.append((X, y, self.a.copy()))

    def reset_batch(self):
        self.batch = []

    def meta_adapt(self, wind=None):
        if len(self.batch) < 50:
            return
        self.inner_adapt_count = 0
        self.meta_adapt_count += 1
        eta_A = self.eta_A_base / np.sqrt(self.meta_adapt_count)

        y_batch = []
        F_batch = []
        self.phi.reset_hidden()
        for X, y, a in self.batch[:-1]:
            x_t = torch.from_numpy(X)
            phi_out = self.phi(x_t, detach_hidden=False)
            kron_mat = torch.kron(torch.eye(3), phi_out)
            y_pred = torch.matmul(kron_mat, torch.from_numpy(a))
            y_batch.append(y_pred)
            F_batch.append(torch.from_numpy(y))
        y_batch = torch.stack(y_batch)
        F_batch = torch.stack(F_batch)

        self.optimizer.zero_grad()
        loss = self.loss(y_batch, F_batch)
        loss.backward()
        self.optimizer.step()

        self.reset_batch()
        self.meta_adapt_count = 0


class MetaAdaptPureSmcDeep(MetaAdaptPureSmc):

    _name = 'omac-pure-smc-deep'
    name_long = 'OMAC-PURE-SMC (deep)'

    class Phi(nn.Module):
        def __init__(self, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(13, layer_sizes[0]))
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def reset_hidden(self):
            pass

        def feature(self, x_np):
            with torch.no_grad():
                phi_out = self(torch.from_numpy(x_np)).numpy()
            return np.kron(np.eye(3), phi_out)

    def __init__(self, dim_a=100, eta_a_base=0.010, eta_A_base=0.005, layer_sizes=(25, 50), train=True, *args, **kwargs):
        super().__init__(train=train, *args, **kwargs)
        # Initialize kernels
        self.layer_sizes = layer_sizes
        self.dim_a = dim_a - dim_a % 3

        # Initialize parameters
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base

        # Initialize learned parameter matrices
        self.a = np.zeros(self.dim_a)
        self.phi = self.Phi(dim_kernel=int(self.dim_a / 3), layer_sizes=self.layer_sizes)
        self.optimizer = optim.Adam(self.phi.parameters(), lr=self.eta_A_base)
        self.loss = nn.MSELoss()

    def reset_controller(self):
        super().reset_controller()

        self.inner_adapt_count = 0
        self.meta_adapt_count = 0

        self.reset_batch()

        self.a = np.zeros(self.dim_a)

        if hasattr(self, 'phi'):
            self.phi.reset_hidden()

    def get_phi(self, X):
        return self.phi.feature(X)

    def get_f_hat(self, X):
        phi = self.get_phi(X)
        return Force(phi @ self.a)

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = max(self.eta_a_base / np.sqrt(self.inner_adapt_count), self.eta_a_base * 0.2)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_phi(X)
        if (np.linalg.norm(self.a)>20):
            self.a = self.a / np.linalg.norm(self.a) * 20
    
    def update_batch(self, X, fhat, y):
        self.batch.append((X, y, self.a.copy()))

    def reset_batch(self):
        self.batch = []

    def meta_adapt(self, wind=None):
        if len(self.batch) < 50:
            return
        self.inner_adapt_count = 0
        self.meta_adapt_count += 1
        eta_A = self.eta_A_base / np.sqrt(self.meta_adapt_count)

        y_batch = []
        F_batch = []
        self.phi.reset_hidden()
        for X, y, a in self.batch[:-1]:
            x_t = torch.from_numpy(X)
            phi_out = self.phi(x_t)
            kron_mat = torch.kron(torch.eye(3), phi_out)
            y_pred = torch.matmul(kron_mat, torch.from_numpy(a))
            y_batch.append(y_pred)
            F_batch.append(torch.from_numpy(y))
        y_batch = torch.stack(y_batch)
        F_batch = torch.stack(F_batch)

        self.optimizer.zero_grad()
        loss = self.loss(y_batch, F_batch)
        loss.backward()
        self.optimizer.step()

        self.reset_batch()
        self.meta_adapt_count = 0


class MetaAdaptPureSmcDeepRnnAtt(MetaAdaptPureSmc):

    _name = 'omac-pure-smc-deepRnnAtt'
    name_long = 'OMAC-PURE-SMC (deepRnnAtt)'

    class Phi(nn.Module):
        def __init__(self, dim_kernel, layer_sizes):
            super().__init__()
            self.fc1 = spectral_norm(nn.Linear(13, layer_sizes[0]))
            self.rnn1 = nn.RNN(layer_sizes[0], layer_sizes[0], 2)
            self.rnn1 = self._sn_rnn(self.rnn1)
            self.fc2 = spectral_norm(nn.Linear(layer_sizes[0], layer_sizes[1]))
            self.rnn2 = nn.RNN(layer_sizes[1], layer_sizes[1], 2)
            self.rnn2 = self._sn_rnn(self.rnn2)
            self.fc3 = spectral_norm(nn.Linear(layer_sizes[1], dim_kernel))
            self.att = LuongAttention(query_dim=layer_sizes[1], key_dim=layer_sizes[0])
            self._h1 = None
            self._h2 = None

        @staticmethod
        def _sn_rnn(rnn_layer):
            for name, _ in list(rnn_layer.named_parameters()):
                if 'weight' in name:
                    spectral_norm(rnn_layer, name)
            return rnn_layer

        def reset_hidden(self):
            self._h1 = None
            self._h2 = None

        def forward(self, x, detach_hidden=True):
            squeeze_out = x.dim() == 1
            if squeeze_out:
                x = x.unsqueeze(0)
            if x.dim() == 2:
                x = x.unsqueeze(0)

            T, B, _ = x.shape
            x_fc1 = F.relu(self.fc1(x.reshape(T * B, -1))).reshape(T, B, -1)
            out1, h1 = self.rnn1(x_fc1, self._h1)
            self._h1 = h1.detach() if detach_hidden else h1

            x_fc2 = F.relu(self.fc2(out1.reshape(T * B, -1))).reshape(T, B, -1)
            out2, h2 = self.rnn2(x_fc2, self._h2)
            self._h2 = h2.detach() if detach_hidden else h2

            ctx = self.att(out2[-1:], self._h1, self._h1)
            feat = out2[-1] + ctx
            out = self.fc3(feat)

            if squeeze_out:
                out = out.squeeze(0)
            return out

        def feature(self, x_np):
            with torch.no_grad():
                phi_out = self(torch.from_numpy(x_np)).numpy()
            return np.kron(np.eye(3), phi_out)

    def __init__(self, dim_a=100, eta_a_base=0.010, eta_A_base=0.005, layer_sizes=(25, 50), train=True, *args, **kwargs):
        super().__init__(train=train, *args, **kwargs)
        # Initialize kernels
        self.layer_sizes = layer_sizes
        self.dim_a = dim_a - dim_a % 3
        
        # Initialize parameters
        self.eta_a_base = eta_a_base
        self.eta_A_base = eta_A_base

        # Initialize learned parameter matrices
        self.a = np.zeros(self.dim_a)
        self.phi = self.Phi(dim_kernel=int(self.dim_a / 3), layer_sizes=self.layer_sizes)
        self.optimizer = optim.Adam(self.phi.parameters(), lr=self.eta_A_base)
        self.loss = nn.MSELoss()
    
    def reset_controller(self):
        super().reset_controller()

        self.inner_adapt_count = 0
        self.meta_adapt_count = 0

        self.reset_batch()

        self.a = np.zeros(self.dim_a)

        if hasattr(self, 'phi'):
            self.phi.reset_hidden()

    def get_phi(self, X):
        return self.phi.feature(X)

    def get_f_hat(self, X):
        phi = self.get_phi(X)
        return Force(phi @ self.a)

    def inner_adapt(self, X, fhat, y):
        self.inner_adapt_count += 1
        eta_a = max(self.eta_a_base / np.sqrt(self.inner_adapt_count), self.eta_a_base * 0.2)
        self.a -= eta_a * 2 * (fhat - y).transpose() @ self.get_phi(X)
        if (np.linalg.norm(self.a)>20):
            self.a = self.a / np.linalg.norm(self.a) * 20

    def update_batch(self, X, fhat, y):
        self.batch.append((X, y, self.a.copy()))

    def reset_batch(self):
        self.batch = []

    def meta_adapt(self, wind=None):
        if len(self.batch) < 50:
            return
        self.inner_adapt_count = 0
        self.meta_adapt_count += 1
        eta_A = self.eta_A_base / np.sqrt(self.meta_adapt_count)

        y_batch = []
        F_batch = []
        self.phi.reset_hidden()
        for X, y, a in self.batch[:-1]:
            x_t = torch.from_numpy(X)
            phi_out = self.phi(x_t, detach_hidden=False)
            kron_mat = torch.kron(torch.eye(3), phi_out)
            y_pred = torch.matmul(kron_mat, torch.from_numpy(a))
            y_batch.append(y_pred)
            F_batch.append(torch.from_numpy(y))
        y_batch = torch.stack(y_batch)
        F_batch = torch.stack(F_batch)

        self.optimizer.zero_grad()
        loss = self.loss(y_batch, F_batch)
        loss.backward()
        self.optimizer.step()

        self.reset_batch()
        self.meta_adapt_count = 0
