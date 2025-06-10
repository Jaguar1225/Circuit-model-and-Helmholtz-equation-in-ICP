import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import scipy as sp

class HelmholtzModel:
    def __init__(self, config: dict):
        """
        헬름홀츠 방정식 모델 초기화
        """
        self.config = config
        self.omega = 2 * np.pi * config['circuit']['radio_frequency_Hz']
        self.mu0 = 4 * np.pi * 1e-7  # 진공 투자율

    def solve(self, sigma_p: np.ndarray, mesh_r: np.ndarray, mesh_z: np.ndarray, coil_current: float) -> np.ndarray:
        """
        헬름홀츠 방정식 풀이
        
        Args:
            sigma_p: 플라즈마 전도도 (복소수 배열)
            mesh_r, mesh_z: 격자 좌표
            coil_current: 코일 전류
            
        Returns:
            E_field: 전기장 (복소수 배열)
        """
        nz, nr = sigma_p.shape
        dr = mesh_r[0, 1] - mesh_r[0, 0]
        dz = mesh_z[1, 0] - mesh_z[0, 0]

        # 계수 행렬 구성
        A = self._build_matrix(sigma_p, mesh_r, mesh_z)
        
        # 우변 벡터 구성
        b = self._build_rhs(mesh_r, mesh_z, coil_current)
        
        # 선형 시스템 풀이
        E = spsolve(A, b)
        
        # 결과를 2D 배열로 변환
        E_field = E.reshape(nz, nr)
        
        return E_field

    def _build_matrix(self, sigma_p: np.ndarray, mesh_r: np.ndarray, mesh_z: np.ndarray) -> sp.sparse.csr_matrix:
        """
        행렬 구성 (수치적 안정성 개선)
        """
        nr, nz = mesh_r.shape
        n = nr * nz
        dr = mesh_r[0, 1] - mesh_r[0, 0]
        dz = mesh_z[1, 0] - mesh_z[0, 0]
        diagonals = []
        offsets = []
        d_center = np.zeros(n, dtype=complex)
        for i in range(nz):
            for j in range(nr):
                k = i * nr + j
                r = np.maximum(mesh_r[i, j], 1e-6)
                s_p = np.maximum(np.abs(sigma_p[i, j]), 1e-10)
                d_center[k] = -2 * (1 / dr**2 + 1 / dz**2) - 1j * self.omega * self.mu0 * s_p
        diagonals.append(d_center)
        offsets.append(0)
        d_r_plus = np.zeros(n, dtype=complex)
        d_r_minus = np.zeros(n, dtype=complex)
        for i in range(nz):
            for j in range(nr):
                k = i * nr + j
                r = np.maximum(mesh_r[i, j], 1e-6)
                d_r_plus[k] = 1 / dr**2 + 1 / (2 * r * dr)
                d_r_minus[k] = 1 / dr**2 - 1 / (2 * r * dr)
        diagonals.append(d_r_plus)
        offsets.append(1)
        diagonals.append(d_r_minus)
        offsets.append(-1)
        d_z_plus = np.ones(n, dtype=complex) / dz**2
        d_z_minus = np.ones(n, dtype=complex) / dz**2
        diagonals.append(d_z_plus)
        offsets.append(nr)
        diagonals.append(d_z_minus)
        offsets.append(-nr)
        A = sp.sparse.diags(diagonals, offsets, format='csr')
        A = A + 1e-10 * sp.sparse.eye(n, dtype=complex)
        return A

    def _build_rhs(self, mesh_r: np.ndarray, mesh_z: np.ndarray, coil_current: float) -> np.ndarray:
        """
        우변 벡터 구성 (수치적 안정성 개선)
        """
        nr, nz = mesh_r.shape
        n = nr * nz
        J_coil = np.zeros(n, dtype=complex)
        coil_r = np.maximum(self.config['circuit']['coil_radius'], 1e-6)
        coil_z = np.maximum(self.config['circuit']['coil_height'], 1e-6)
        for i in range(nz):
            for j in range(nr):
                k = i * nr + j
                r = np.maximum(mesh_r[i, j], 1e-6)
                z = np.maximum(mesh_z[i, j], 1e-6)
                if (abs(r - coil_r) < 0.1 * coil_r and abs(z - coil_z) < 0.1 * coil_z):
                    J_coil[k] = coil_current
        return J_coil 