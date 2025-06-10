import numpy as np
import math
from typing import Tuple
from scipy.sparse import eye
import multiprocessing as mp
from functools import partial
import os

class PlasmaModel:
    def __init__(self, config: dict):
        '''
        Plasma model (최적화된 버전)

        Args:
            config: Configuration dictionary
        '''
        self.config = config
        self.num_cores = max(1, os.cpu_count() - 1)  # CPU 코어 수 - 1
        self.pool = None  # 필요할 때 초기화
        
        # 캐시된 계산 결과
        self._cache = {}
        self._cache_size = 1000  # 최대 캐시 크기

    def _parallel_calculate(self, func, *args):
        """병렬 계산을 수행하는 헬퍼 함수"""
        # 입력 데이터의 크기 확인
        data_size = len(args[0])
        chunk_size = max(1, data_size // self.num_cores)
        
        # 청크 단위로 데이터 분할
        chunks = []
        for i in range(0, data_size, chunk_size):
            chunk_end = min(i + chunk_size, data_size)
            chunk_args = tuple(arg[i:chunk_end] for arg in args)
            chunks.append(chunk_args)
        
        # 각 청크에 대해 계산 수행
        results = []
        for chunk in chunks:
            chunk_result = func(*chunk)
            # 결과가 튜플인 경우 각 요소의 차원 확인
            if isinstance(chunk_result, tuple):
                results.append(chunk_result)
            else:
                results.append(chunk_result)
        
        # 결과 병합
        if isinstance(results[0], tuple):
            # 튜플의 각 요소별로 병합
            merged_results = []
            for i in range(len(results[0])):
                merged = np.concatenate([r[i] for r in results])
                merged_results.append(merged)
            return tuple(merged_results)
        else:
            return np.concatenate(results)

    def _cache_result(self, key, func, *args, **kwargs):
        """계산 결과 캐싱"""
        if key in self._cache:
            return self._cache[key]
        
        result = func(*args, **kwargs)
        
        # 캐시 크기 관리
        if len(self._cache) >= self._cache_size:
            # 가장 오래된 항목 제거
            self._cache.pop(next(iter(self._cache)))
        
        self._cache[key] = result
        return result

    def solve_continuity_equation(
            self, 
            ng: np.ndarray, ne: np.ndarray, ni: np.ndarray, nms: np.ndarray, 
            Te: np.ndarray,
            E_field: np.ndarray,
            dt: float,
            dr: float,
            dz: float
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Solve continuity equation

        Args:
            ng: Neutral gas density
            ne: Electron density
            ni: Ion density
            nms: Metastable density
            Te: Electron temperature
            E_field: Electric field
            dt: Time step size
            dr, dz: Grid spacing

        Returns:
            tuple of (ne, ni, ng, nms): Updated densities
        '''
        ne, ni, ng, nms = self.cal_reactions(Te, ne, ng, nms, ni)
        j_e, j_i, j_g, j_ms = self.cal_transport(ne, ng, nms, ni, E_field, dr, dz)

        ne = ne + j_e * dt
        ni = ni + j_i * dt
        ng = ng + j_g * dt
        nms = nms + j_ms * dt

        return ne, ni, ng, nms

    def cal_transport(
            self,
            ne: np.ndarray, ng: np.ndarray, nms: np.ndarray, ni: np.ndarray,
            E_field: np.ndarray,
            dr: float,
            dz: float
            ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], 
                      tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        '''
        Calculate transport fluxes in r and z directions

        Args:
            ne: Electron density
            ng: Neutral gas density
            nms: Metastable density
            ni: Ion density
            E_field: Electric field (complex)
            dr, dz: Grid spacing

        Returns:
            tuple of ((j_e_r, j_e_z), (j_i_r, j_i_z), (j_g_r, j_g_z), (j_ms_r, j_ms_z)):
            Transport fluxes for electrons, ions, neutrals, and metastables in r and z directions
        '''
        D_e = float(self.config['chemistry']['transport']['D_e'])
        D_g = float(self.config['chemistry']['transport']['D_g'])
        D_ms = float(self.config['chemistry']['transport']['D_ms'])
        D_i = float(self.config['chemistry']['transport']['D_i'])
        mu_e = float(self.config['chemistry']['transport']['mu_e'])
        mu_i = float(self.config['chemistry']['transport']['mu_i'])

        # Gradient calculation (using NumPy gradient)
        grad_ne_r, grad_ne_z = np.gradient(ne, dr, dz)
        grad_ni_r, grad_ni_z = np.gradient(ni, dr, dz)
        grad_ng_r, grad_ng_z = np.gradient(ng, dr, dz)
        grad_nms_r, grad_nms_z = np.gradient(nms, dr, dz)

        # Electric field components (assuming E_field is complex)
        E_r = np.real(E_field)
        E_z = np.imag(E_field)

        # Diffusion fluxes (r and z components)
        diff_e_r = -D_e * grad_ne_r
        diff_e_z = -D_e * grad_ne_z
        diff_i_r = -D_i * grad_ni_r
        diff_i_z = -D_i * grad_ni_z
        diff_g_r = -D_g * grad_ng_r
        diff_g_z = -D_g * grad_ng_z
        diff_ms_r = -D_ms * grad_nms_r
        diff_ms_z = -D_ms * grad_nms_z

        # Drift fluxes (r and z components)
        drift_e_r = -mu_e * E_r * ne
        drift_e_z = -mu_e * E_z * ne
        drift_i_r = -mu_i * E_r * ni
        drift_i_z = -mu_i * E_z * ni

        # Total fluxes (r and z components)
        j_e_r = diff_e_r + drift_e_r
        j_e_z = diff_e_z + drift_e_z
        j_i_r = diff_i_r + drift_i_r
        j_i_z = diff_i_z + drift_i_z
        j_g_r = diff_g_r
        j_g_z = diff_g_z
        j_ms_r = diff_ms_r
        j_ms_z = diff_ms_z

        return (j_e_r, j_e_z), (j_i_r, j_i_z), (j_g_r, j_g_z), (j_ms_r, j_ms_z)

    def _calculate_chunk(self, Te, ne, ni, ng):
        """청크 단위로 충돌 주파수를 계산하는 함수"""
        # 전자-중성 입자 충돌 주파수 계산
        nu_en = self.calculate_electron_neutral_collision_frequency(Te, ne, ng)
        
        # 전자-이온 충돌 주파수 계산
        nu_ei = self.calculate_electron_ion_collision_frequency(Te, ne, ni)
        
        return nu_en, nu_ei

    def calculate_collision_frequencies(self, Te, ne, ni, ng):
        """충돌 주파수를 계산하는 메서드 (병렬 처리 지원)"""
        # 캐시 키 생성
        cache_key = (hash(Te.tobytes()), hash(ne.tobytes()), 
                    hash(ni.tobytes()), hash(ng.tobytes()))
        
        return self._cache_result(cache_key, self._parallel_calculate, 
                                self._calculate_chunk, Te, ne, ni, ng)

    def cal_heating(self, Te: np.ndarray, ne: np.ndarray, ng: np.ndarray, ni: np.ndarray, E_r: np.ndarray, E_z: np.ndarray, A_theta: np.ndarray, t: float, dt: float) -> np.ndarray:
        '''전자 가열 계산 (개선된 버전)'''
        # 1. 입력값 수치적 안정성 보강
        Te = np.nan_to_num(Te, nan=0.1, posinf=100.0, neginf=0.1)
        Te = np.clip(Te, 0.1, 100.0)  # Te 범위 제한
        ne = np.nan_to_num(ne, nan=1e6, posinf=1e20, neginf=1e6)
        ne = np.clip(ne, 1e6, 1e20)  # ne 범위 제한
        ng = np.nan_to_num(ng, nan=1e18, posinf=1e25, neginf=1e18)
        ng = np.clip(ng, 1e18, 1e25)  # ng 범위 제한
        ni = np.nan_to_num(ni, nan=1e6, posinf=1e20, neginf=1e6)
        ni = np.clip(ni, 1e6, 1e20)  # ni 범위 제한
        
        # 2. 전기장 수치적 안정성 보강
        E_r = np.nan_to_num(E_r, nan=0.0, posinf=1e6, neginf=-1e6)
        E_z = np.nan_to_num(E_z, nan=0.0, posinf=1e6, neginf=-1e6)
        E_r = np.clip(E_r, -1e6, 1e6)
        E_z = np.clip(E_z, -1e6, 1e6)
        A_theta = np.nan_to_num(A_theta, nan=0.0, posinf=1e3, neginf=-1e3)
        A_theta = np.clip(A_theta, -1e3, 1e3)
        
        # 3. 가열 항 계산 (수치적 안정성 개선)
        omega = 2 * np.pi * self.config['process']['radio_frequency_Hz']
        e = self.config['constants']['e']
        me = self.config['constants']['me']
        eps0 = self.config['constants']['eps0']
        mu0 = self.config['constants']['mu0']
        
        # 3.1 오믹 가열
        nu_en, nu_ei = self.calculate_collision_frequencies(Te, ne, ng, ni)
        nu_m = nu_en + nu_ei
        nu_m = np.clip(nu_m, 1e6, 1e12)  # 충돌 주파수 범위 제한
        
        # 전기장 크기 계산 (수치적 안정성 개선)
        E_mag = np.sqrt(np.abs(E_r)**2 + np.abs(E_z)**2 + np.abs(1j * omega * A_theta)**2)
        E_mag = np.clip(E_mag, 1e-10, 1e6)  # 전기장 크기 범위 제한
        
        # 오믹 가열 계산
        P_ohm = (e**2 * ne * E_mag**2) / (me * (nu_m**2 + omega**2))
        P_ohm = np.clip(P_ohm, 0.0, 1e10)  # 가열률 범위 제한
        
        # 3.2 확률론적 가열 (수치적 안정성 개선)
        v_e = np.sqrt(2 * e * Te / me)  # 전자 열속도
        v_e = np.clip(v_e, 1e3, 1e7)  # 전자 속도 범위 제한
        
        # 플라즈마 전도도 계산 (수치적 안정성)
        sigma_p = self.calculate_plasma_conductivity(Te, ne, ni, ng)
        sigma_p = np.clip(np.nan_to_num(sigma_p, nan=1e-10, posinf=1e8, neginf=1e-10), 1e-10, 1e8)
        
        # 스킨 깊이 계산
        skin_depth = np.sqrt(2 / (mu0 * omega * sigma_p))
        skin_depth = np.clip(skin_depth, 1e-4, 1e-1)  # 스킨 깊이 범위 제한
        
        # 확률론적 가열 계산
        P_stoch = 0.25 * ne * me * v_e**3 / skin_depth
        P_stoch = np.clip(P_stoch, 0.0, 1e10)  # 가열률 범위 제한
        
        # 3.3 공진 가열 (수치적 안정성 개선)
        P_res = np.zeros_like(Te)
        resonance_mask = np.abs(nu_m - omega) < 0.1 * omega
        P_res[resonance_mask] = 0.5 * P_ohm[resonance_mask]
        
        # 4. 냉각 항 계산 (수치적 안정성 개선)
        # 4.1 충돌 냉각
        mi = self.config['constants']['mi']
        P_coll = 2 * me * nu_m * ne * Te / (mi * e)
        P_coll = np.clip(P_coll, 0.0, 1e10)  # 냉각률 범위 제한
        
        # 4.2 이온화/여기 손실
        k_iz = self.calculate_ionization_rate(Te)
        k_ex = self.calculate_excitation_rate(Te)
        E_iz = 15.76  # 아르곤 이온화 에너지 [eV]
        E_ex = 11.5   # 아르곤 여기 에너지 [eV]
        
        P_ion = k_iz * ne * ng * E_iz
        P_ex = k_ex * ne * ng * E_ex
        P_ion = np.clip(P_ion, 0.0, 1e10)
        P_ex = np.clip(P_ex, 0.0, 1e10)
        
        # 4.3 벽 손실
        v_B = np.sqrt(e * Te / (2 * np.pi * me))  # 봄 속도
        v_B = np.clip(v_B, 1e2, 1e5)  # 봄 속도 범위 제한
        
        P_wall = 0.25 * ne * v_B * e * Te / self.config['chamber']['chamber_radius_m']
        P_wall = np.clip(P_wall, 0.0, 1e10)  # 벽 손실 범위 제한
        
        # 5. 순 가열률 계산
        P_net = P_ohm + P_stoch + P_res - P_coll - P_ion - P_ex - P_wall
        P_net = np.clip(P_net, -1e10, 1e10)  # 순 가열률 범위 제한
        
        # 6. 전자 온도 변화 계산 (수치적 안정성 개선)
        dTe = (2/3) * P_net * dt / (ne * e)
        
        # 7. 동적 제약 조건 적용 (온도 변화 제한)
        dTe_max = np.where(Te < 1.0, 0.5, 0.1) * Te  # 저온에서 더 큰 변화 허용
        dTe = np.clip(dTe, -dTe_max, dTe_max)
        
        # 8. 새로운 온도 계산 및 범위 제한
        Te_new = Te + dTe
        Te_new = np.clip(Te_new, 0.1, 100.0)  # 최종 온도 범위 제한
        
        # 9. 경계 조건 적용
        # 외부 경계: 완전 반사 조건
        Te_new[0, :] = Te_new[1, :]  # 상단 경계
        Te_new[-1, :] = Te_new[-2, :]  # 하단 경계
        Te_new[:, 0] = Te_new[:, 1]  # 좌측 경계
        Te_new[:, -1] = Te_new[:, -2]  # 우측 경계
        
        # 내부 경계: 대칭 조건 (r=0)
        Te_new[:, 0] = Te_new[:, 1]  # r=0에서 대칭
        
        return Te_new

    def cal_reactions(
            self,
            Te: np.ndarray,
            ne: np.ndarray, ng: np.ndarray, nms: np.ndarray, ni: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Calculate reaction rates and update densities

        Args:
            Te: Electron temperature
            ne: Electron density
            ng: Neutral gas density
            nms: Metastable density
            ni: Ion density

        Returns:
            tuple of (ne, ni, ng, nms): Updated densities
        '''
        Te = np.maximum(Te, 0.1)
        ne = np.maximum(np.abs(ne), 1e6)
        ni = np.maximum(np.abs(ni), 1e6)
        ng = np.maximum(np.abs(ng), 1e10)
        nms = np.maximum(np.abs(nms), 1e6)
        kB = self.config['constants']['k']
        kiz = self.config['chemistry']['reactions']['ionization']['k0'] * \
              np.exp(-self.config['chemistry']['reactions']['ionization']['E_ion_J'] / 
                    (kB * np.maximum(Te, 0.1)))
        kstep = self.config['chemistry']['reactions']['step_ionization']['k0'] * \
                np.exp(-self.config['chemistry']['reactions']['step_ionization']['E_step_J'] / 
                      (kB * np.maximum(Te, 0.1)))
        kex = self.config['chemistry']['reactions']['excitation']['k0'] * \
              np.exp(-self.config['chemistry']['reactions']['excitation']['E_exc_J'] / 
                    (kB * np.maximum(Te, 0.1)))
        kre = self.config['chemistry']['reactions']['recombination']['radiative']['k0'] * \
              np.exp(-self.config['chemistry']['reactions']['recombination']['radiative']['E_re_J'] / 
                    (kB * np.maximum(Te, 0.1)))
        kri = self.config['chemistry']['reactions']['recombination']['three_body']['k0'] * \
              np.exp(-self.config['chemistry']['reactions']['recombination']['three_body']['E_re_J'] / 
                    (kB * np.maximum(Te, 0.1)))
        dne = kiz * ng * ne + kstep * ne * nms - kre * ne**2 * ni - kri * ne * ni
        dni = kiz * ng * ne + kstep * ne * nms - kre * ni * ne - kri * ni * ne**2
        dng = -kiz * ng * ne - kex * ne * ng + kre * ni * ne + kri * ne**2 * ni
        dms = kex * ne * ng - kstep * ne * nms
        ne = ne + dne
        ni = ni + dni
        ng = ng + dng
        nms = nms + dms
        ne = np.maximum(np.abs(ne), 1e6)
        ni = np.maximum(np.abs(ni), 1e6)
        ng = np.maximum(np.abs(ng), 1e10)
        nms = np.maximum(np.abs(nms), 1e6)
        return ne, ni, ng, nms
    
    def solve(self, ng: np.ndarray, ne: np.ndarray, ni: np.ndarray, nms: np.ndarray,
              Te: np.ndarray, E_field: np.ndarray, dt: float, dr: float, dz: float) -> Tuple[np.ndarray, ...]:
        '''
        플라즈마 방정식 풀이 (최적화된 벡터화 버전)
        '''
        # 1. 안전한 실수 변환 및 클리핑 (한 번에 처리)
        Te = np.nan_to_num(Te, nan=0.1, posinf=100.0, neginf=0.1)
        Te = np.clip(Te, 0.1, 100.0)  # [eV]
        ne = np.nan_to_num(ne, nan=1e6, posinf=1e20, neginf=1e6)
        ne = np.clip(ne, 1e6, 1e20)   # [m^-3]
        ng = np.nan_to_num(ng, nan=1e6, posinf=1e20, neginf=1e6)
        ng = np.clip(ng, 1e6, 1e20)   # [m^-3]
        # ni도 마찬가지로 처리
        if 'ni' in locals() or 'ni' in globals():
            ni = np.nan_to_num(ni, nan=1e6, posinf=1e20, neginf=1e6)
            ni = np.clip(ni, 1e6, 1e20)
        E_field = np.abs(np.real(E_field))
        
        # 2. 반응률 계산 (벡터화 및 캐싱)
        kB = self.config['constants']['k']
        Te_safe = np.maximum(Te, 0.1)  # 안전한 Te 값
        
        # 반응률 계산 (벡터화)
        kiz = self.calculate_ionization_rate(Te_safe)
        kex = self.calculate_excitation_rate(Te_safe)
        kstep = self.calculate_step_ionization_rate(Te_safe)
        kre = self.calculate_recombination_rate(Te_safe)
        
        # 3. 소스항 계산 (벡터화)
        S_ion = kiz * ng * ne
        S_ex = kex * ne * ng
        S_step = kstep * ne * nms
        S_rec = kre * ne * ni
        
        # 4. 전송 플럭스 계산 (드리프트 + 확산)
        j_e, j_i, j_g, j_ms = self.cal_transport(ne, ng, nms, ni, E_field, dr, dz)
        
        # 5. 플럭스 발산 계산
        div_j_e_r = np.gradient(j_e[0], dr)[0]  # r 방향
        div_j_e_z = np.gradient(j_e[1], dz)[0]  # z 방향
        div_j_e = div_j_e_r + div_j_e_z
        
        div_j_i_r = np.gradient(j_i[0], dr)[0]
        div_j_i_z = np.gradient(j_i[1], dz)[0]
        div_j_i = div_j_i_r + div_j_i_z
        
        div_j_g_r = np.gradient(j_g[0], dr)[0]
        div_j_g_z = np.gradient(j_g[1], dz)[0]
        div_j_g = div_j_g_r + div_j_g_z
        
        div_j_ms_r = np.gradient(j_ms[0], dr)[0]
        div_j_ms_z = np.gradient(j_ms[1], dz)[0]
        div_j_ms = div_j_ms_r + div_j_ms_z
        
        # 6. 밀도 업데이트 (반응 + 전송)
        dne = S_ion + S_step - S_rec - div_j_e
        dni = S_ion + S_step - S_rec - div_j_i
        dng = -S_ion - S_ex + S_rec - div_j_g
        dnms = S_ex - S_step - div_j_ms
        
        # 7. 전자 온도 업데이트 (벡터화)
        nu_en = self.calculate_collision_frequencies(Te, ne, ni, ng)[0]
        nu_ei = self.calculate_collision_frequencies(Te, ne, ni, ng)[1]
        sigma_p = self.calculate_plasma_conductivity(Te, ne, ni, ng)
        sigma_p = np.clip(np.nan_to_num(sigma_p, nan=1e-10, posinf=1e8, neginf=1e-10), 1e-10, 1e8)
        
        P_ohm = sigma_p * E_field**2
        Q_e_g = nu_en * ne * kB * Te
        Q_e_i = nu_ei * ne * kB * Te
        
        denom = np.clip(1.5 * ne * kB, 1e-10, 1e10)
        dTe = (P_ohm - Q_e_g - Q_e_i) / denom
        
        # 8. 변수 업데이트 (안정화된 버전)
        ne_new = np.clip(ne + dne * dt, 1e6, 1e25)
        ni_new = np.clip(ni + dni * dt, 1e6, 1e25)
        ng_new = np.clip(ng + dng * dt, 1e10, 1e25)
        nms_new = np.clip(nms + dnms * dt, 1e6, 1e25)
        Te_new = np.clip(Te + dTe * dt, 0.1, 100.0)
        
        # 9. NaN 및 무한대 처리
        ne_new = np.nan_to_num(ne_new, nan=1e10, posinf=1e25, neginf=1e6)
        ni_new = np.nan_to_num(ni_new, nan=1e10, posinf=1e25, neginf=1e6)
        ng_new = np.nan_to_num(ng_new, nan=1e10, posinf=1e25, neginf=1e10)
        nms_new = np.nan_to_num(nms_new, nan=1e6, posinf=1e25, neginf=1e6)
        Te_new = np.nan_to_num(Te_new, nan=0.1, posinf=100.0, neginf=0.1)
        
        return ne_new, ni_new, ng_new, nms_new, Te_new

    def calculate_ohmic_heating(self, ne: np.ndarray, Te: np.ndarray, E_field: np.ndarray) -> np.ndarray:
        '''Joule heating 계산 (벡터화)'''
        sigma = self.calculate_plasma_conductivity(Te, ne, ne, ne)  # ne ≈ ni
        return sigma * np.abs(E_field)**2

    def calculate_cooling(self, ne: np.ndarray, ni: np.ndarray, 
                         ng: np.ndarray, nms: np.ndarray, Te: np.ndarray) -> np.ndarray:
        '''Cooling 계산 (벡터화)'''
        # 전자-이온 충돌 냉각
        nu_ei = self.calculate_electron_ion_collision_frequency(ne, ni, Te)
        P_ei = 3 * ne * Te * nu_ei

        # 전자-중성 충돌 냉각
        nu_en = self.calculate_electron_neutral_collision_frequency(ne, ng, Te)
        P_en = 3 * ne * Te * nu_en

        # 여기/이온화 냉각 (eV 단위)
        k_iz = self.calculate_ionization_rate(Te)
        k_ex = self.calculate_excitation_rate(Te)
        E_iz = 15.76  # 아르곤 이온화 에너지 [eV]
        E_ex = 11.5   # 아르곤 여기 에너지 [eV]
        P_iz = k_iz * ne * ng * E_iz
        P_ex = k_ex * ne * ng * E_ex

        return P_ei + P_en + P_iz + P_ex

    def calculate_plasma_conductivity(self, Te: np.ndarray, ne: np.ndarray, ni: np.ndarray, ng: np.ndarray) -> np.ndarray:
        e = self.config['constants']['e']
        me = self.config['constants']['me']
        eps0 = self.config['constants']['eps0']

        # 안전한 실수 변환
        Te = np.real(np.clip(Te, 0.1, 100.0))
        ne = np.real(np.clip(ne, 1e6, 1e25))
        ni = np.real(np.clip(ni, 1e6, 1e25))
        ng = np.real(np.clip(ng, 1e6, 1e25))
        
        # 열 속도
        k_B = self.config['constants']['k_B_eV']
        v_th = np.sqrt(np.clip(8 * k_B * Te / (np.pi * me), 1e-10, 1e10))
        sigma_en = 1e-19
        nu_en = ng * sigma_en * v_th
        nu_en = np.clip(nu_en, 1e6, 1e15)
        ln_Lambda = 10
        v_th3 = np.clip(v_th**3, 1e-20, 1e20)
        nu_ei = ni * (e**4 * ln_Lambda) / (4 * np.pi * eps0**2 * me**2 * v_th3)
        nu_ei = np.clip(np.real(nu_ei), 1e6, 1e15)
        nu_tot = nu_en + nu_ei
        nu_tot = np.clip(nu_tot, 1e6, 1e15)
        sigma_p = ne * e**2 / (me * nu_tot)
        sigma_p = np.clip(np.real(sigma_p), 1e-10, 1e8)
        sigma_p = np.nan_to_num(sigma_p, nan=1e-10, posinf=1e8, neginf=1e-10)
        return sigma_p

    def calculate_ionization_rate(self, Te: np.ndarray) -> np.ndarray:
        """개선된 이온화 반응률 계산 (수치적 안정성 강화)
        
        Args:
            Te: 전자 온도 [eV]
            
        Returns:
            이온화 반응률 [m^3/s]
        """
        Te_safe = np.maximum(Te, 0.1)  # 최소 온도 적용
        kiz = 2.34e-14 * np.sqrt(Te_safe) * np.exp(-13.6 / Te_safe)
        return np.clip(kiz, 0, 1e-12)  # 과도한 반응률 방지

    def calculate_step_ionization_rate(self, Te: np.ndarray) -> np.ndarray:
        """개선된 단계 이온화 반응률 계산 (수치적 안정성 강화)

        Args:
            Te: 전자 온도 [eV]

        Returns:
            단계 이온화 반응률 [m^3/s]
        """
        Te_safe = np.maximum(Te, 0.1)  # 최소 온도 적용
        kstep = 1.5e-13 * np.sqrt(Te_safe) * np.exp(-4.0 / Te_safe)
        return np.clip(kstep, 0, 1e-12)  # 과도한 반응률 방지

    def calculate_excitation_rate(self, Te: np.ndarray) -> np.ndarray:
        """개선된 여기 반응률 계산 (수치적 안정성 강화)

        Args:
            Te: 전자 온도 [eV]

        Returns:
            여기 반응률 [m^3/s]
        """
        Te_safe = np.maximum(Te, 0.1)  # 최소 온도 적용
        k0 = 1e-13  # 기본 반응률 계수
        E_a = 11.5  # 여기 에너지 [eV]
        k_ex = k0 * np.exp(-E_a / Te_safe)
        return np.clip(k_ex, 0, 1e-12)  # 과도한 반응률 방지

    def calculate_deexcitation_rate(self, Te: np.ndarray) -> np.ndarray:
        # 임시로 여기의 1/10로 설정 (실험적)
        k_ex = self.calculate_excitation_rate(Te)
        k_dex = 0.1 * k_ex
        return k_dex

    def calculate_recombination_rate(self, Te: np.ndarray) -> np.ndarray:
        # radiative recombination 사용
        k0 = self.config['chemistry']['reactions']['recombination']['radiative']['k0']
        E_a = self.config['chemistry']['reactions']['recombination']['radiative']['E_a']
        k_rec = k0 * np.exp(-E_a / np.maximum(Te, 1e-2))
        return k_rec

    def calculate_electron_diffusion(self, Te: np.ndarray) -> np.ndarray:
        # transport 파라미터 사용
        D_e = self.config['chemistry']['transport']['D_e']
        return np.ones_like(Te) * D_e

    def calculate_ion_diffusion(self, Te: np.ndarray) -> np.ndarray:
        D_i = self.config['chemistry']['transport']['D_i']
        return np.ones_like(Te) * D_i

    def calculate_electron_velocity(self, Te: np.ndarray, E_field: np.ndarray) -> np.ndarray:
        e = self.config['constants']['e']
        me = self.config['constants']['me']
        Te = np.real(np.clip(Te, 0.1, 100.0))
        E_field = np.real(E_field)
        nu_tot = self.calculate_electron_neutral_collision_frequency(
            np.ones_like(Te), np.ones_like(Te), Te
        ) + self.calculate_electron_ion_collision_frequency(
            np.ones_like(Te), np.ones_like(Te), Te
        )
        nu_tot = np.clip(nu_tot, 1e6, 1e15)
        mu_e = e / (me * nu_tot)
        mu_e = np.clip(np.real(mu_e), 1e-10, 1e2)
        v_e = mu_e * E_field
        v_e = np.clip(np.real(v_e), -1e7, 1e7)
        v_e = np.nan_to_num(v_e, nan=0.0, posinf=1e7, neginf=-1e7)
        return v_e

    def calculate_ion_velocity(self, Te: np.ndarray, E_field: np.ndarray) -> np.ndarray:
        e = self.config['constants']['e']
        mi = self.config['constants']['mi']
        Te = np.real(np.clip(Te, 0.1, 100.0))
        E_field = np.real(E_field)
        k_B = self.config['constants']['k_B_eV']
        v_th_i = np.sqrt(np.clip(8 * k_B * Te / (np.pi * mi), 1e-10, 1e10))
        sigma_in = 1e-18
        nu_in = np.ones_like(Te) * sigma_in * v_th_i
        nu_in = np.clip(np.real(nu_in), 1e6, 1e15)
        mu_i = e / (mi * nu_in)
        mu_i = np.clip(np.real(mu_i), 1e-10, 1e2)
        v_i = mu_i * E_field
        v_i = np.clip(np.real(v_i), -1e5, 1e5)
        v_i = np.nan_to_num(v_i, nan=0.0, posinf=1e5, neginf=-1e5)
        return v_i

    def calculate_electron_ion_collision_frequency(self, ne: np.ndarray, ni: np.ndarray, Te: np.ndarray) -> np.ndarray:
        e = self.config['constants']['e']
        me = self.config['constants']['me']
        eps0 = self.config['constants']['eps0']
        k_B = self.config['constants']['k_B_eV']
        Te = np.real(np.clip(Te, 0.1, 100.0))
        ni = np.real(np.clip(ni, 1e6, 1e25))
        v_th = np.sqrt(np.clip(8 * k_B * Te / (np.pi * me), 1e-10, 1e10))
        ln_Lambda = 10
        v_th3 = np.clip(v_th**3, 1e-20, 1e20)
        nu_ei = ni * (e**4 * ln_Lambda) / (4 * np.pi * eps0**2 * me**2 * v_th3)
        nu_ei = np.clip(np.real(nu_ei), 1e6, 1e15)
        nu_ei = np.nan_to_num(nu_ei, nan=1e6, posinf=1e15, neginf=1e6)
        return nu_ei

    def calculate_electron_neutral_collision_frequency(self, ne: np.ndarray, ng: np.ndarray, Te: np.ndarray) -> np.ndarray:
        me = self.config['constants']['me']
        k_B = self.config['constants']['k_B_eV']
        Te = np.real(np.clip(Te, 0.1, 100.0))
        ng = np.real(np.clip(ng, 1e6, 1e25))
        v_th = np.sqrt(np.clip(8 * k_B * Te / (np.pi * me), 1e-10, 1e10))
        sigma_en = 1e-19
        nu_en = ng * sigma_en * v_th
        nu_en = np.clip(np.real(nu_en), 1e6, 1e15)
        nu_en = np.nan_to_num(nu_en, nan=1e6, posinf=1e15, neginf=1e6)
        return nu_en

    def __del__(self):
        """소멸자: 멀티프로세싱 풀 정리"""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
    
    