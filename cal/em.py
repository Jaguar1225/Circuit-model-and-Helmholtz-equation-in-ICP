import numpy as np
import math
from .mesh import Mesh
from scipy.sparse import lil_matrix, csr_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from typing import Tuple
import time

class EMModel:
    def __init__(self, config: dict, mesh: Mesh):
        '''
        EM model

        Args:
            config: Configuration dictionary
            mesh: Mesh 객체
        '''
        self.config = config
        self.mesh = mesh  # numpy mesh 사용
        self.mesh_r = mesh.mesh_r
        self.mesh_z = mesh.mesh_z
        self.nr = mesh.mesh_r.shape[1]
        self.nz = mesh.mesh_z.shape[0]
        
        # 격자 간격
        self.dr = mesh.mesh_r[0,1] - mesh.mesh_r[0,0]
        self.dz = mesh.mesh_z[1,0] - mesh.mesh_z[0,0]
        
        # 행렬 구성에 필요한 인덱스 배열 미리 계산
        self._setup_matrix_indices()

    def _setup_matrix_indices(self):
        """행렬 구성에 필요한 인덱스 배열을 미리 계산"""
        n = self.nr * self.nz
        self.indices = np.arange(n).reshape(self.nz, self.nr)
        
        # 내부 점들의 인덱스 (경계 제외)
        self.inner_indices = self.indices[1:-1, 1:-1].flatten()
        
        # 이웃 점들의 인덱스 (r, z 방향)
        self.right_indices = self.indices[1:-1, 2:].flatten()
        self.left_indices = self.indices[1:-1, :-2].flatten()
        self.up_indices = self.indices[2:, 1:-1].flatten()
        self.down_indices = self.indices[:-2, 1:-1].flatten()
        
        # 중심점 인덱스
        self.center_indices = self.indices[1:-1, 1:-1].flatten()

    def set_coil_current(self, I: float) -> np.ndarray:
        '''코일 전류 밀도 설정 (벡터화)'''
        I_peak = I * math.sqrt(2)
        coil_r = self.config['circuit']['coil_radius']
        coil_z = self.config['circuit']['coil_height']
        
        # 코일 위치 근처의 격자점 찾기 (벡터화)
        r_dist = np.abs(self.mesh_r - coil_r)
        z_dist = np.abs(self.mesh_z - coil_z)
        coil_mask = (r_dist < 0.1 * coil_r) & (z_dist < 0.1 * coil_z)
        
        J_coil = np.zeros((self.nz, self.nr), dtype=complex)
        J_coil[coil_mask] = I_peak
        
        return J_coil

    def solve_helmholtz_equation(self, sigma_p: np.ndarray, J_coil: np.ndarray, ne: np.ndarray, ni: np.ndarray) -> np.ndarray:
        '''헬름홀츠 방정식 풀이 (플라즈마 전하 분포 포함)'''
        omega = 2 * math.pi * self.config['process']['radio_frequency_Hz']
        mu0 = self.config['constants']['mu0']
        e = self.config['constants']['e']
        eps0 = self.config['constants']['eps0']
        
        # 입력값 수치적 안정성 보강
        sigma_p = np.nan_to_num(sigma_p, nan=1e-10, posinf=1e8, neginf=1e-10)
        sigma_p = np.clip(sigma_p, 1e-10, 1e8)
        ne = np.nan_to_num(ne, nan=1e6, posinf=1e20, neginf=1e6)
        ne = np.clip(ne, 1e6, 1e20)
        ni = np.nan_to_num(ni, nan=1e6, posinf=1e20, neginf=1e6)
        ni = np.clip(ni, 1e6, 1e20)
        
        # 1. 행렬 구성 (벡터화)
        n = self.nr * self.nz
        
        # 중심 대각선 계수
        r_vals = np.maximum(self.mesh_r, 1e-6)  # r = 0 방지
        s_p_vals = np.maximum(np.abs(sigma_p), 1e-10)  # 0 방지
        
        # 플라즈마 전하 분포에 의한 항 추가
        charge_density = e * (ni - ne)  # 전하 밀도
        charge_density = np.nan_to_num(charge_density, nan=0.0, posinf=1e20, neginf=-1e20)
        plasma_term = -1j * omega * mu0 * charge_density / (eps0 * omega**2)
        
        # 경계에서의 완전 반사 조건을 위한 계수
        boundary_coeff = 1e-3 * 1j * omega * mu0 * s_p_vals
        
        diag_coeff = -2 * (1/self.dr**2 + 1/self.dz**2) - 1j * omega * mu0 * s_p_vals + plasma_term
        
        # 행렬 구성
        A = lil_matrix((n, n), dtype=complex)
        
        # 내부 점들에 대한 계수 설정 (벡터화)
        A[self.center_indices, self.center_indices] = diag_coeff[1:-1, 1:-1].flatten()
        A[self.center_indices, self.right_indices] = 1/self.dr**2 + 1/(2 * r_vals[1:-1, 1:-1].flatten() * self.dr)
        A[self.center_indices, self.left_indices] = 1/self.dr**2 - 1/(2 * r_vals[1:-1, 1:-1].flatten() * self.dr)
        A[self.center_indices, self.up_indices] = 1/self.dz**2
        A[self.center_indices, self.down_indices] = 1/self.dz**2
        
        # 2. 우변 벡터 구성 (벡터화)
        b = np.zeros(n, dtype=complex)
        # 코일 전류 항
        b[self.center_indices] = mu0 * r_vals[1:-1, 1:-1].flatten() * J_coil[1:-1, 1:-1].flatten()
        
        # 플라즈마 전하 분포에 의한 항 추가
        grad_charge_r, grad_charge_z = np.gradient(charge_density, self.dr, self.dz)
        grad_charge_r = np.nan_to_num(grad_charge_r, nan=0.0, posinf=1e20, neginf=-1e20)
        grad_charge_z = np.nan_to_num(grad_charge_z, nan=0.0, posinf=1e20, neginf=-1e20)
        
        b[self.center_indices] += -1j * omega * mu0 * r_vals[1:-1, 1:-1].flatten() * (
            grad_charge_r[1:-1, 1:-1].flatten() + grad_charge_z[1:-1, 1:-1].flatten()
        ) / (eps0 * omega**2)
        
        # 3. 경계 조건 적용
        # 외부 경계: 완전 반사 조건 (Robin 경계 조건)
        # z 경계
        z_boundary_indices_top = self.indices[0, :]
        z_boundary_indices_bottom = self.indices[-1, :]
        A[z_boundary_indices_top, z_boundary_indices_top] = 1 + boundary_coeff[0, :]
        A[z_boundary_indices_top, self.indices[1, :]] = -1
        A[z_boundary_indices_bottom, z_boundary_indices_bottom] = 1 + boundary_coeff[-1, :]
        A[z_boundary_indices_bottom, self.indices[-2, :]] = -1
        
        # r 경계
        r_boundary_indices_left = self.indices[:, 0]
        r_boundary_indices_right = self.indices[:, -1]
        A[r_boundary_indices_left, r_boundary_indices_left] = 1 + boundary_coeff[:, 0]
        A[r_boundary_indices_left, self.indices[:, 1]] = -1
        A[r_boundary_indices_right, r_boundary_indices_right] = 1 + boundary_coeff[:, -1]
        A[r_boundary_indices_right, self.indices[:, -2]] = -1
        
        # 내부 경계: 강화된 대칭 경계 조건 (r=0)
        symmetry_indices = self.indices[1:-1, 0]
        A[symmetry_indices, symmetry_indices] = 2  # 대칭성 강화
        A[symmetry_indices, self.indices[1:-1, 1]] = -2
        b[symmetry_indices] = 0
        
        # 4. 행렬 변환 및 풀이
        A = A.tocsr()
        A = A + 1e-5 * eye(n, dtype=complex)  # regularization(특이성 완화, 더 강하게)
        t0 = time.time()
        x = spsolve(A, b)
        t1 = time.time()
        print(f"[Helmholtz] spsolve 시간: {t1-t0:.3f}s")
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("[Helmholtz] spsolve 결과에 NaN/Inf 발생!")
        A_theta = x.reshape(self.nz, self.nr)
        
        # 6. 결과 검증 및 후처리
        A_theta = np.nan_to_num(A_theta, nan=0.0, posinf=1e3, neginf=-1e3)
        A_theta = np.clip(A_theta, -1e3, 1e3)
        
        return A_theta

    def calculate_electric_field(self, A_theta: np.ndarray) -> np.ndarray:
        '''전기장 계산 (벡터화)'''
        omega = 2 * math.pi * self.config['process']['radio_frequency_Hz']
        return -1j * omega * A_theta
