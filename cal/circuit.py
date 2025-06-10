import numpy as np
import math
from typing import Tuple

class CircuitModel:
    def __init__(self, config: dict):
        '''
        Circuit model

        Args:
            config: Configuration dictionary
        '''
        self.config = config
        self.Rp = 0
        self.omega = 2 * np.pi * config['circuit']['radio_frequency_Hz']
        self.mu0 = config['constants']['mu0']  # 진공 투자율

        r_max = self.config['coil']['coil_radius_max']
        r_min = self.config['coil']['coil_radius_min']
        coil_width = self.config['coil']['coil_width']
        N_turns = self.config['coil']['number_of_turns']

        self.r_coil = [r_min + i * coil_width + (2*i+1)*coil_width/2 for i in range(N_turns)]

        self.coil_length = sum([2*math.pi*r for r in self.r_coil])
        self.Rc = config['coil']['coil_resistance_ohm'] * self.coil_length / coil_width**2 / math.pi

    def calculate_plasma_resistance(self, sigma_p: np.ndarray, mesh_r: np.ndarray, mesh_z: np.ndarray) -> float:
        '''플라즈마 저항 계산 (벡터화)'''
        # 격자 간격
        dr = mesh_r[0,1] - mesh_r[0,0]
        dz = mesh_z[1,0] - mesh_z[0,0]
        
        # 적분 가중치 (2D 격자)
        r_weights = np.ones_like(mesh_r) * dr
        z_weights = np.ones_like(mesh_z) * dz
        r_weights[:,0] = dr/2  # r=0 경계
        r_weights[:,-1] = dr/2  # r=r_max 경계
        z_weights[0,:] = dz/2  # z=0 경계
        z_weights[-1,:] = dz/2  # z=z_max 경계
        
        # 2D 적분 가중치
        weights = r_weights * z_weights
        
        # 전도도 적분 (벡터화)
        sigma_integral = np.sum(sigma_p * weights)
        
        # 플라즈마 저항 계산
        Rp = 1 / (2 * math.pi * sigma_integral)
        
        return Rp

    def calculate_power_dissipation(self, sigma_p: np.ndarray, E_theta: np.ndarray, mesh_r: np.ndarray, mesh_z: np.ndarray) -> Tuple[float, float]:
        '''전력 소산 계산 (벡터화)'''
        # 격자 간격
        dr = mesh_r[0,1] - mesh_r[0,0]
        dz = mesh_z[1,0] - mesh_z[0,0]
        
        # 적분 가중치 (2D 격자)
        r_weights = np.ones_like(mesh_r) * dr
        z_weights = np.ones_like(mesh_z) * dz
        r_weights[:,0] = dr/2  # r=0 경계
        r_weights[:,-1] = dr/2  # r=r_max 경계
        z_weights[0,:] = dz/2  # z=0 경계
        z_weights[-1,:] = dz/2  # z=z_max 경계
        
        # 2D 적분 가중치
        weights = r_weights * z_weights
        
        # 전기장 제곱 (벡터화)
        E_squared = np.abs(E_theta) ** 2
        
        # 전도도 적분 (벡터화)
        sigma_integral = np.sum(sigma_p * weights)
        
        # 전력 소산 (벡터화)
        P_plasma = np.sum(sigma_p * E_squared * weights) * math.pi
        
        # 코일 전력 소산 (벡터화)
        P_coil = self.Rc * (self.I_peak / math.sqrt(2)) ** 2
        
        return P_plasma, P_coil

    def calculate_efficiency(
            self,
            sigma_p: np.ndarray,
            E_field: np.ndarray,
            mesh_r: np.ndarray,
            mesh_z: np.ndarray,
            input_power: float
            ) -> float:
        """
        전력 효율 계산
        
        Args:
            sigma_p: 플라즈마 전도도 (복소수 배열)
            E_field: 전기장 (복소수 배열)
            mesh_r, mesh_z: 격자 좌표
            input_power: 입력 전력
            
        Returns:
            eta: 전력 효율 (0~1 사이의 실수)
        """
        P_diss = self.calculate_power_dissipation(sigma_p, E_field, mesh_r, mesh_z)
        eta = P_diss / input_power if input_power > 0 else 0.0
        return float(min(max(eta, 0.0), 1.0))  # 0~1 사이로 제한
    
    def calculate_coil_current(self, t: float) -> float:
        '''
        Calculate coil current

        Returns:
            I: Coil current (scalar)
        '''
        # Rp가 이미 스칼라 값이므로 직접 계산 가능
        total_resistance = self.Rp + self.Rc
        if total_resistance <= 0:
            return 1e-10
        I_peak = np.sqrt(self.config['process']['input_power_W'] / total_resistance)
        omega = self.omega
        return I_peak * np.sin(omega * t)