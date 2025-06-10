import numpy as np
from fipy import CellVariable
from typing import Tuple, Dict, List, Optional
from .mesh import Mesh
from .circuit import CircuitModel
from .em import EMModel
from .plasma import PlasmaModel
from .plot import PlasmaPlotter
import os
from tqdm import tqdm
from datetime import datetime
import time
import gc

class SelfConsistentSolver:
    def __init__(self, plasma_model: PlasmaModel, circuit_model: CircuitModel, em_model: EMModel,
                 mesh: Mesh, config: dict):
        """
        자기 일관성 해결기 초기화
        
        Args:
            plasma_model: 플라즈마 모델
            circuit_model: 회로 모델
            em_model: 전자기 모델
            mesh: 메시 객체
            config: 설정 딕셔너리
        """
        self.plasma_model = plasma_model
        self.circuit_model = circuit_model
        self.em_model = em_model
        self.mesh = mesh
        self.config = config
        
        # 수렴 설정 최적화
        self.tolerance = 1e-2  # 수렴 허용 오차 완화 (1e-3 -> 1e-2)
        self.relaxation_factor = 0.4  # relaxation factor 증가 (0.3 -> 0.4)
        self.max_iterations = 50  # 최대 반복 횟수 감소 (100 -> 50)
        
        # 적응형 relaxation을 위한 변수
        self.prev_error = float('inf')
        self.relaxation_history = []
        self.error_history = []
        
        # 결과 저장 경로
        self.results_dir = os.path.join('results', datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 플로터 초기화
        self.mesh_obj = mesh
        self.r_axis = mesh.mesh_r[0, :]
        self.z_axis = mesh.mesh_z[:, 0]
        self.plotter = PlasmaPlotter(mesh, self.r_axis, self.z_axis)
        
        # 플롯 저장 간격 증가
        self.plot_interval = 200  # 200 스텝마다 플롯 저장 (100 -> 200)
        
        self.ne = mesh.ne
        self.ni = mesh.ni
        self.ng = mesh.ng
        self.nms = mesh.nms
        self.Te = mesh.Te
        self.E_field = mesh.E_field
        self.Rp = mesh.Rp
        
        # 변수 초기화
        self._initialize_variables(mesh)
        
        # 이전 스텝의 해를 저장할 변수들
        self._initialize_previous_variables()
        
        # 시간 의존성 추적을 위한 변수들
        self._initialize_time_history()
        
        # 수렴 이력 추적
        self.convergence_history = []
        self.current_iteration = 0
        
        # 병렬 처리를 위한 멀티프로세싱 풀 초기화
        self.num_cores = max(1, os.cpu_count() - 1)  # CPU 코어 수 - 1
        self.pool = None  # 필요할 때 초기화

    def _initialize_variables(self, mesh: Mesh) -> None:
        """변수 초기화"""
        self.ne = mesh.ne.copy()
        self.ni = mesh.ni.copy()
        self.ng = mesh.ng.copy()
        self.nms = mesh.nms.copy()
        self.Te = mesh.Te.copy()
        self.E_field = mesh.E_field.copy()
        self.Rp = mesh.Rp
        self.current_time = 0.0
        self.time_step = 0.0

    def _initialize_previous_variables(self) -> None:
        """이전 스텝 변수 초기화"""
        self.ne_prev = self.ne.copy()
        self.ni_prev = self.ni.copy()
        self.ng_prev = self.ng.copy()
        self.nms_prev = self.nms.copy()
        self.Te_prev = self.Te.copy()
        self.E_field_prev = self.E_field.copy()

    def _initialize_time_history(self) -> None:
        """시간 이력 변수 초기화"""
        self.time_history = {
            'time': [],  # 시간
            'ne_avg': [], 'ni_avg': [], 'ng_avg': [], 'nms_avg': [],
            'Te_avg': [], 'E_avg': [], 'Rp': [], 'I_coil': [],
            'convergence': [],  # 수렴 이력
            
            # 에너지 보존 관련
            'P_ohm': [], 'P_stoch': [], 'P_res': [],  # 가열 항
            'P_coll': [], 'P_ion': [], 'P_ex': [], 'P_wall': [],  # 냉각 항
            'P_net': [],  # 순 에너지 변화
            
            # 엔트로피 생성 관련
            'S_coll': [], 'S_ion': [], 'S_ex': [], 'S_rec': [],  # 엔트로피 생성 항
            'S_total': []  # 총 엔트로피 생성
        }

    def update_time_history(self) -> None:
        """시간 이력 업데이트"""
        self.time_history['time'].append(self.current_time)
        self.time_history['ne_avg'].append(np.mean(self.ne))
        self.time_history['ni_avg'].append(np.mean(self.ni))
        self.time_history['ng_avg'].append(np.mean(self.ng))
        self.time_history['nms_avg'].append(np.mean(self.nms))
        self.time_history['Te_avg'].append(np.mean(self.Te))
        self.time_history['E_avg'].append(np.mean(np.abs(self.E_field)))
        self.time_history['Rp'].append(self.Rp)
        self.time_history['I_coil'].append(self.circuit_model.calculate_coil_current(self.current_time))
        
        # 에너지 보존 데이터 업데이트
        E_r, E_z, A_theta = self.solve_electric_field()
        E_field = E_r + 1j * E_z
        E_mag = np.sqrt(np.abs(E_r)**2 + np.abs(E_z)**2 + np.abs(1j * self.circuit_model.omega * A_theta)**2)
        
        # 가열 항
        nu_en, nu_ei = self.plasma_model.calculate_collision_frequencies(self.Te, self.ne, self.ni, self.ng)
        nu_m = nu_en + nu_ei
        e = self.config['constants']['e']
        me = self.config['constants']['me']
        
        P_ohm = (e**2 * self.ne * E_mag**2) / (me * (nu_m**2 + self.circuit_model.omega**2))
        v_e = np.sqrt(2 * e * self.Te / me)
        skin_depth = np.sqrt(2 / (self.config['constants']['mu0'] * self.circuit_model.omega * 
                                self.plasma_model.calculate_plasma_conductivity(self.Te, self.ne, self.ni, self.ng)))
        P_stoch = 0.25 * self.ne * me * v_e**3 / skin_depth
        P_res = np.zeros_like(P_ohm)
        resonance_mask = np.abs(nu_m - self.circuit_model.omega) < 0.1 * self.circuit_model.omega
        P_res[resonance_mask] = 0.5 * P_ohm[resonance_mask]
        
        # 냉각 항
        mi = self.config['constants']['mi']
        P_coll = 2 * me * nu_m * self.ne * self.Te / (mi * e)
        k_iz = self.plasma_model.calculate_ionization_rate(self.Te)
        k_ex = self.plasma_model.calculate_excitation_rate(self.Te)
        E_iz = 15.76  # 아르곤 이온화 에너지 [eV]
        E_ex = 11.5   # 아르곤 여기 에너지 [eV]
        P_ion = k_iz * self.ne * self.ng * E_iz
        P_ex = k_ex * self.ne * self.ng * E_ex
        v_B = np.sqrt(e * self.Te / (2 * np.pi * me))
        P_wall = 0.25 * self.ne * v_B * e * self.Te / self.config['chamber']['chamber_radius_m']
        
        # 순 에너지 변화
        P_net = P_ohm + P_stoch + P_res - P_coll - P_ion - P_ex - P_wall
        
        # 에너지 데이터 저장
        self.time_history['P_ohm'].append(np.mean(P_ohm))
        self.time_history['P_stoch'].append(np.mean(P_stoch))
        self.time_history['P_res'].append(np.mean(P_res))
        self.time_history['P_coll'].append(np.mean(P_coll))
        self.time_history['P_ion'].append(np.mean(P_ion))
        self.time_history['P_ex'].append(np.mean(P_ex))
        self.time_history['P_wall'].append(np.mean(P_wall))
        self.time_history['P_net'].append(np.mean(P_net))
        
        # 엔트로피 생성 계산
        kB = self.config['constants']['k']
        T = np.clip(self.Te * e / kB, 1e-2, None)  # 0.01K 이상으로 강제
        P_coll = np.nan_to_num(P_coll, nan=0.0, posinf=0.0, neginf=0.0)
        P_ion = np.nan_to_num(P_ion, nan=0.0, posinf=0.0, neginf=0.0)
        P_ex = np.nan_to_num(P_ex, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 충돌 엔트로피
        S_coll = P_coll / T
        
        # 이온화 엔트로피
        S_ion = P_ion / T
        
        # 여기 엔트로피
        S_ex = P_ex / T
        
        # 재결합 엔트로피
        k_rec = self.plasma_model.calculate_recombination_rate(self.Te)
        E_iz = self.config['chemistry']['reactions']['ionization']['E_ion_J'] / self.config['constants']['e']  # 아르곤 이온화 에너지 [eV]
        P_rec = k_rec * self.ne * self.ni * E_iz
        P_rec = np.nan_to_num(P_rec, nan=0.0, posinf=0.0, neginf=0.0)
        S_rec = P_rec / T
        
        # 총 엔트로피 생성
        S_total = S_coll + S_ion + S_ex + S_rec
        
        # 엔트로피 데이터 저장
        self.time_history['S_coll'].append(np.mean(S_coll))
        self.time_history['S_ion'].append(np.mean(S_ion))
        self.time_history['S_ex'].append(np.mean(S_ex))
        self.time_history['S_rec'].append(np.mean(S_rec))
        self.time_history['S_total'].append(np.mean(S_total))

    def solve_step(self, t: float, dt: float) -> Tuple[bool, Tuple[CellVariable, ...]]:
        '''한 시간 스텝 계산'''
        # 1. 초기화
        max_iter = 200  # 최대 반복 횟수 증가
        tol = 1e-2  # 수렴 기준 완화
        relax = 0.5  # 초기 이완 계수 증가
        
        # 2. 이전 값 저장
        old_vars = (self.ne, self.ni, self.ng, self.nms, self.Te)
        
        # 3. 반복 계산
        with tqdm(total=max_iter, desc='내부 반복', leave=False) as pbar:
            for iter in range(max_iter):
                t0 = time.time()
                E_r, E_z, A_theta = self.solve_electric_field()
                t1 = time.time()
                Te_new = self.plasma_model.cal_heating(
                    self.Te, self.ne, self.ng, self.ni,
                    E_r, E_z, A_theta, t, dt
                )
                t2 = time.time()
                
                # 3.2 플라즈마 변수 업데이트
                # 전자 가열 계산 (수정된 인자 전달)
                E_field = E_r + 1j * E_z
                drift_e, drift_i, drift_g, drift_ms = self.plasma_model.cal_transport(
                    self.ne, self.ng, self.nms, self.ni, E_field, self.mesh_obj.dr, self.mesh_obj.dz
                )
                drift_e_r, drift_e_z = drift_e
                drift_i_r, drift_i_z = drift_i
                
                # 반응률 계산
                k_iz = self.plasma_model.calculate_ionization_rate(self.Te)
                k_ex = self.plasma_model.calculate_excitation_rate(self.Te)
                k_step = self.plasma_model.calculate_step_ionization_rate(self.Te)
                k_rec = self.plasma_model.calculate_recombination_rate(self.Te)
                
                # 플럭스 발산 계산
                div_flux_e = self.cal_flux_divergence((drift_e_r, drift_e_z))
                div_flux_i = self.cal_flux_divergence((drift_i_r, drift_i_z))
                
                # 밀도 변화 계산
                dne = dt * (
                    k_iz * self.ne * self.ng +  # 이온화
                    k_step * self.ne * self.ng -  # 단계 이온화
                    k_rec * self.ne * self.ni -  # 재결합
                    div_flux_e  # 전자 수송
                )
                
                dni = dt * (
                    k_iz * self.ne * self.ng +  # 이온화
                    k_step * self.ne * self.ng -  # 단계 이온화
                    k_rec * self.ne * self.ni -  # 재결합
                    div_flux_i  # 이온 수송
                )
                
                dng = -dt * (
                    k_iz * self.ne * self.ng +  # 이온화
                    k_step * self.ne * self.ng  # 단계 이온화
                )
                
                dnm = dt * (
                    k_ex * self.ne * self.ng -  # 여기
                    self.plasma_model.calculate_deexcitation_rate(self.Te) * self.nms * self.ng  # 소멸
                )
                
                # 3.3 이완 적용
                self.Te = (1 - relax) * self.Te + relax * Te_new
                self.ne = np.maximum(0, (1 - relax) * self.ne + relax * (self.ne + dne))
                self.ni = np.maximum(0, (1 - relax) * self.ni + relax * (self.ni + dni))
                self.ng = np.maximum(0, (1 - relax) * self.ng + relax * (self.ng + dng))
                self.nms = np.maximum(0, (1 - relax) * self.nms + relax * (self.nms + dnm))
                
                # 3.4 수렴 검사
                rel_diff = {
                    'ne': np.max(np.abs(self.ne - old_vars[0]) / np.maximum(old_vars[0], 1e6)),
                    'ni': np.max(np.abs(self.ni - old_vars[1]) / np.maximum(old_vars[1], 1e6)),
                    'ng': np.max(np.abs(self.ng - old_vars[2]) / np.maximum(old_vars[2], 1e18)),
                    'Te': np.max(np.abs(self.Te - old_vars[3]) / np.maximum(old_vars[3], 0.1)),
                    'nm': np.max(np.abs(self.nms - old_vars[4]) / np.maximum(old_vars[4], 1e6))
                }
                
                max_rel_diff = max(rel_diff.values())
                
                # 3.5 동적 이완 계수 조정
                if iter > 0:
                    if max_rel_diff > self.prev_error:
                        relax *= 0.5  # 발산 시 이완 계수 감소
                    else:
                        relax = min(0.9, relax * 1.1)  # 수렴 시 이완 계수 증가
                
                self.prev_error = max_rel_diff
                
                # 3.6 수렴 확인
                if max_rel_diff < tol:
                    pbar.update(max_iter - iter)  # 남은 반복 횟수만큼 업데이트
                    return True, (self.ne, self.ni, self.ng, self.nms, self.Te)
                
                # 3.7 이전 값 업데이트
                old_vars = (self.ne, self.ni, self.ng, self.nms, self.Te)
                
                # NaN/Inf 체크
                for name, arr in zip(
                    ['Te', 'ne', 'ni', 'ng', 'nms', 'E_r', 'E_z', 'A_theta'],
                    [self.Te, self.ne, self.ni, self.ng, self.nms, E_r, E_z, A_theta]
                ):
                    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                        print(f"[경고] {name}에서 NaN 또는 Inf 발생! 반복 {iter}에서 중단")
                        break
                print(f"[{iter}] solve_electric_field: {t1-t0:.3f}s, cal_heating: {t2-t1:.3f}s")
            
        return False, (self.ne, self.ni, self.ng, self.nms, self.Te)

    def _update_previous_variables(self) -> None:
        """이전 스텝 변수 업데이트"""
        self.ne_prev = self.ne.copy()
        self.ni_prev = self.ni.copy()
        self.ng_prev = self.ng.copy()
        self.nms_prev = self.nms.copy()
        self.Te_prev = self.Te.copy()
        self.E_field_prev = self.E_field.copy()

    def _calculate_adaptive_relaxation(self, old: np.ndarray, new: np.ndarray) -> float:
        """적응형 relaxation factor 계산"""
        # 기본 relaxation factor
        w_base = self.relaxation_factor
        
        # 상대 변화량 계산
        rel_diff = np.abs((new - old) / (np.abs(old) + 1e-10))
        max_rel_diff = np.max(rel_diff)
        
        # 변화가 클수록 relaxation factor 감소
        if max_rel_diff > 1.0:
            w = w_base * 0.5
        elif max_rel_diff > 0.5:
            w = w_base * 0.7
        elif max_rel_diff > 0.1:
            w = w_base * 0.9
        else:
            w = w_base
        
        return w

    def check_convergence(self, old_vars: Tuple[CellVariable, ...], 
                         new_vars: Tuple[CellVariable, ...]) -> bool:
        '''
        수렴 판정: 상대 오차와 절대 오차를 모두 고려 (개선된 버전)
        '''
        rel_tol = self.tolerance
        abs_tol_density = 1e10  # 밀도 절대 오차 기준
        abs_tol_temp = 0.1      # 온도 절대 오차 기준
        
        # 각 변수별 수렴 판정
        converged = True
        max_rel_diff = 0.0
        
        for i, (old, new) in enumerate(zip(old_vars, new_vars)):
            # 상대 오차 계산
            rel_diff = np.abs((new - old) / (np.abs(old) + 1e-10))
            max_rel_diff = max(max_rel_diff, np.max(rel_diff))
            
            # 절대 오차 계산
            abs_diff = np.abs(new - old)
            
            # 변수 타입에 따른 수렴 판정
            if i < 4:  # 밀도 변수 (ne, ni, ng, nms)
                if np.any(rel_diff > rel_tol) and np.any(abs_diff > abs_tol_density):
                    converged = False
                    break
            else:  # 온도 변수 (Te)
                if np.any(rel_diff > rel_tol) and np.any(abs_diff > abs_tol_temp):
                    converged = False
                    break
        
        # 수렴 이력 저장
        self.convergence_history.append(max_rel_diff)
        self.time_history['convergence'].append(max_rel_diff)
        
        # 수렴 상태 출력 (디버깅) - 100회 반복마다 출력하도록 수정
        if self.current_iteration % 100 == 0:
            print(f"\nIteration {self.current_iteration}:")
            print(f"Max relative difference: {max_rel_diff:.2e}")
            print(f"Converged: {converged}")
        
        return converged

    def solve(self, total_time: float, dt: float, 
             plot_interval: Optional[int] = None,
             save_dir: Optional[str] = None) -> Dict[str, List[float]]:
        '''
        주어진 총 시간 동안 시뮬레이션을 수행하고 결과를 시각화 (최적화된 버전)

        Args:
            total_time: 총 시뮬레이션 시간
            dt: 시간 스텝 크기
            plot_interval: 시각화 간격 (스텝 단위, 기본값: 200)
            save_dir: 결과 저장 디렉토리 (기본값: "results")

        Returns:
            시간에 따른 모든 변수들의 이력
        '''
        if plot_interval is not None:
            self.plot_interval = plot_interval
        if save_dir is not None:
            self.results_dir = save_dir
            
        os.makedirs(self.results_dir, exist_ok=True)
        self.plotter = PlasmaPlotter(self.mesh, self.r_axis, self.z_axis)
        
        n_steps = int(total_time / dt)
        self.current_time = 0.0
        
        # 초기 상태 저장 및 시각화
        self.update_time_history()
        self._save_distributions(step=0)
        
        print(f"시뮬레이션 시작: {n_steps} 스텝, dt = {dt:.2e}s")
        print(f"결과 저장 경로: {self.results_dir}")
        print(f"수렴 설정: 허용 오차 = {self.tolerance:.1e}, 이완 계수 = {self.relaxation_factor:.2f}")
        print(f"병렬 처리: {self.num_cores} 코어 사용")
        print("진행 상황과 수렴 정보는 200회 반복마다 출력됩니다")
        
        # 메모리 최적화를 위한 변수
        last_save_time = time.time()
        save_interval = 60  # 1분마다 중간 결과 저장
        
        try:
            for step in tqdm(range(n_steps), desc="진행 상황"):
                # 각 시간 스텝에서의 수렴 계산
                converged = False
                for iter in range(self.max_iterations):
                    old_vars = (self.ne, self.ni, self.ng, self.nms, self.Te)
                    converged, new_vars = self.solve_step(self.current_time, dt)
                    
                    if converged and self.check_convergence(old_vars, new_vars):
                        break
                
                if not converged:
                    print(f"\n경고: 시간 스텝 {step}이(가) {self.max_iterations}회 반복 후에도 수렴하지 않았습니다")
                    print(f"현재 상대 오차: {self.convergence_history[-1]:.2e}")
                
                # 시간 업데이트
                self.current_time += dt
                self.current_iteration += 1
                
                # 주기적인 결과 저장 및 시각화
                if step % self.plot_interval == 0:
                    self._save_distributions(step)
                
                # 메모리 관리를 위한 중간 결과 저장
                current_time = time.time()
                if current_time - last_save_time > save_interval:
                    self._save_intermediate_results()
                    last_save_time = current_time
                
                # 진행 상황 출력 (200회 반복마다)
                if self.current_iteration % 200 == 0:
                    print(f"\n반복 {self.current_iteration}:")
                    print(f"최대 상대 오차: {self.convergence_history[-1]:.2e}")
                    print(f"수렴 상태: {converged}")
                    print(f"현재 시간: {self.current_time:.2e}s")
        
        except KeyboardInterrupt:
            print("\n사용자에 의해 시뮬레이션이 중단되었습니다.")
            print("현재까지의 결과를 저장합니다...")
            self._save_intermediate_results()
            raise
        
        finally:
            # 최종 결과 저장
            self._save_final_results()
            if self.pool is not None:
                self.pool.close()
                self.pool.join()
        
        return self.time_history

    def _save_distributions(self, step: int) -> None:
        """현재 상태의 공간 분포를 저장"""
        if self.plotter is None:
            return
            
        # 파일명에 시간 정보 포함
        time_str = f"t{self.current_time:.2e}".replace('.', 'p')
        
        # 밀도 분포 저장
        self.plotter.plot_density_distribution(
            self.ne, self.ni, self.ng, self.nms,
            title=f"Density Distribution (t = {self.current_time:.2e}s)",
            save_path=os.path.join(self.results_dir, f"density_{time_str}.png")
        )
        
        # 온도 분포 저장
        self.plotter.plot_temperature_distribution(
            self.Te,
            title=f"Electron Temperature (t = {self.current_time:.2e}s)",
            save_path=os.path.join(self.results_dir, f"temperature_{time_str}.png")
        )
        
        # 전기장 분포 저장
        self.plotter.plot_electric_field_distribution(
            self.E_field,
            title=f"Electric Field (t = {self.current_time:.2e}s)",
            save_path=os.path.join(self.results_dir, f"field_{time_str}.png")
        )
        
        # 반경 방향 프로파일 저장 (z = 0.05m에서)
        self.plotter.plot_radial_profiles(
            self.ne, self.Te, self.E_field, z_pos=0.05,
            title=f"Radial Profiles at z = 0.05m (t = {self.current_time:.2e}s)",
            save_path=os.path.join(self.results_dir, f"radial_profiles_{time_str}.png")
        )
        
        # 축 방향 프로파일 저장 (r = 0.02m에서)
        self.plotter.plot_axial_profiles(
            self.ne, self.Te, self.E_field, r_pos=0.02,
            title=f"Axial Profiles at r = 0.02m (t = {self.current_time:.2e}s)",
            save_path=os.path.join(self.results_dir, f"axial_profiles_{time_str}.png")
        )

    def _save_final_results(self) -> None:
        """최종 결과 저장 및 시각화"""
        # 시간 이력 데이터 저장
        np.savez(
            os.path.join(self.results_dir, 'time_history.npz'),
            **self.time_history
        )
        
        # 수렴 이력 저장
        np.savez(
            os.path.join(self.results_dir, 'convergence_history.npz'),
            convergence=self.convergence_history
        )
        
        # 시간 이력 시각화
        if self.plotter is not None:
            # 시간 정보 업데이트
            self.plotter.current_time = self.current_time
            self.plotter.time_step = self.time_step
            
            # 시간 이력 플롯
            self.plotter.plot_time_evolution(
                self.time_history,
                title=f"Time Evolution of Plasma Parameters (t = {self.current_time:.3e}s)",
                save_path=os.path.join(self.results_dir, 'time_evolution.png')
            )
            
            # 에너지 보존 플롯
            self.plotter.plot_energy_conservation(
                self.time_history,
                title=f"Energy Conservation in Plasma (t = {self.current_time:.3e}s)",
                save_path=os.path.join(self.results_dir, 'energy_conservation.png')
            )
            
            # 엔트로피 생성 플롯
            self.plotter.plot_entropy_generation(
                self.time_history,
                title=f"Entropy Generation in Plasma (t = {self.current_time:.3e}s)",
                save_path=os.path.join(self.results_dir, 'entropy_generation.png')
            )
            
            # 최종 상태 분포 플롯
            self._save_distributions(step=int(self.current_time / self.time_step))
        
        print(f"\nSimulation completed. Results saved in: {self.results_dir}")
        print("Final values:")
        print(f"  Electron density: {np.mean(self.ne):.2e} m^-3")
        print(f"  Electron temperature: {np.mean(self.Te):.2f} eV")
        print(f"  Plasma resistance: {self.Rp:.2f} Ohm")
        print(f"  Coil current: {self.circuit_model.calculate_coil_current(self.current_time):.2f} A")
        print("\nEnergy Conservation:")
        print(f"  Net power: {self.time_history['P_net'][-1]:.2e} W")
        print(f"  Total entropy generation: {self.time_history['S_total'][-1]:.2e} J/K/s")

    def get_results(self) -> dict:
        '''
        Get simulation results

        Returns:
            dict: Dictionary containing all simulation results
        '''
        return {
            'Te': self.Te,
            'ne': self.ne,
            'ni': self.ni,
            'ng': self.ng,
            'nms': self.nms,
            'E_field': self.E_field,
            'Rp': self.Rp,
            'I_coil': self.circuit_model.calculate_coil_current(self.current_time)
        }

    def get_resistance_statistics(self) -> dict:
        """
        저항 비교에 대한 통계 정보 반환
        
        Returns:
            dict: 저항 비교 통계 정보
        """
        circuit_Rp = np.array(self.resistance_history['circuit_Rp'])
        actual_Rp = np.array(self.resistance_history['actual_Rp'])
        ratio = circuit_Rp / (actual_Rp + 1e-10)
        
        stats = {
            'mean_ratio': np.mean(ratio),
            'std_ratio': np.std(ratio),
            'max_ratio': np.max(ratio),
            'min_ratio': np.min(ratio),
            'mean_circuit_Rp': np.mean(circuit_Rp),
            'mean_actual_Rp': np.mean(actual_Rp),
            'mean_efficiency': np.mean(self.resistance_history['efficiency'])*100,
            'std_efficiency': np.std(self.resistance_history['efficiency'])*100
        }
        
        return stats

    def cal_flux_divergence(self, j: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        2차원 플럭스(j)의 발산(∇·j)을 중심차분으로 계산합니다.
        j: (r, z) 격자에 정의된 플럭스 (r, z 방향 튜플)
        반환: 발산값 (r, z 격자)
        """
        # j는 (j_r, j_z) 튜플로 전달됨
        j_r, j_z = j
        dr = self.mesh_obj.dr
        dz = self.mesh_obj.dz
        # r, z 격자 크기
        nr, nz = j_r.shape
        div = np.zeros_like(j_r)
        # 내부 격자에 대해 중심차분 적용
        div[1:-1, 1:-1] = (
            (j_r[2:, 1:-1] - j_r[:-2, 1:-1]) / (2 * dr) +
            (j_z[1:-1, 2:] - j_z[1:-1, :-2]) / (2 * dz)
        )
        # 경계는 0으로 둠 (Neumann)
        return div

    def solve_electric_field(self):
        '''전기장 계산: 플라즈마 전도도 → 코일 전류 밀도 → 헬름홀츠 방정식 → 전기장'''
        # 1. 플라즈마 전도도 계산
        sigma_p = self.plasma_model.calculate_plasma_conductivity(self.Te, self.ne, self.ni, self.ng)
        # 2. 코일 전류 밀도 계산
        J_coil = self.em_model.set_coil_current(self.circuit_model.calculate_coil_current(self.current_time))
        # 3. 헬름홀츠 방정식 풀이
        A_theta = self.em_model.solve_helmholtz_equation(sigma_p, J_coil, self.ne, self.ni)
        # 4. 전기장 계산 (E_field: 복소수 2D 배열)
        E_field = self.em_model.calculate_electric_field(A_theta)
        # 실수부/허수부로 분리
        E_r = np.real(E_field)
        E_z = np.imag(E_field)
        return E_r, E_z, A_theta

    def _save_intermediate_results(self) -> None:
        """중간 결과 저장 (메모리 최적화)"""
        # 시간 이력 데이터 저장
        np.savez_compressed(
            os.path.join(self.results_dir, f'time_history_step_{self.current_iteration}.npz'),
            **{k: np.array(v) for k, v in self.time_history.items()}
        )
        
        # 수렴 이력 저장
        np.savez_compressed(
            os.path.join(self.results_dir, f'convergence_history_step_{self.current_iteration}.npz'),
            convergence=np.array(self.convergence_history)
        )
        
        # 메모리 정리
        gc.collect()
