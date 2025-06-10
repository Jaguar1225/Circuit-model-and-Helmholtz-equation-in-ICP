import numpy as np
import matplotlib.pyplot as plt
from fipy import CellVariable
from typing import Tuple, Optional, Dict, List
import matplotlib.colors as colors

class PlasmaPlotter:
    def __init__(self, mesh, r_axis: np.ndarray, z_axis: np.ndarray):
        """
        플라즈마 변수들의 공간 분포를 시각화하는 클래스

        Args:
            mesh: FiPy 메시 객체
            r_axis: 반경 방향 격자점 배열
            z_axis: 축 방향 격자점 배열
        """
        self.mesh = mesh
        self.r_axis = r_axis
        self.z_axis = z_axis
        self.nr = len(r_axis)
        self.nz = len(z_axis)
        self.current_time = 0.0  # 현재 시간 추가
        self.time_step = 0.0     # 시간 스텝 추가
        
        # 컬러맵 설정
        self.cmap_density = 'viridis'  # 밀도용 컬러맵
        self.cmap_temperature = 'hot'   # 온도용 컬러맵
        self.cmap_field = 'RdBu_r'      # 전기장용 컬러맵

    def _reshape_to_2d(self, var):
        if isinstance(var, np.ndarray):
            if var.ndim == 2:
                return var
            else:
                return var.reshape(self.nr, self.nz)
        # FiPy CellVariable 등은 지원하지 않음 (numpy만)
        return var

    def plot_density_distribution(self, 
                                ne: CellVariable,
                                ni: Optional[CellVariable] = None,
                                ng: Optional[CellVariable] = None,
                                nms: Optional[CellVariable] = None,
                                title: str = "Plasma Density Distribution",
                                save_path: Optional[str] = None) -> None:
        """
        밀도 분포를 2D 컬러맵으로 표시

        Args:
            ne: 전자 밀도
            ni: 이온 밀도 (선택)
            ng: 중성 가스 밀도 (선택)
            nms: 준안정 상태 밀도 (선택)
            title: 그래프 제목
            save_path: 저장할 파일 경로 (선택)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)
        
        def get_vmin_vmax(data):
            """데이터의 vmin과 vmax를 계산"""
            positive_data = data[data > 0]
            if len(positive_data) == 0:
                return 1e10, 1e11  # 기본값 설정
            return np.min(positive_data), np.max(data)
        
        # 전자 밀도
        ne_data = np.abs(self._reshape_to_2d(ne))
        vmin, vmax = get_vmin_vmax(ne_data)
        im0 = axes[0,0].pcolormesh(self.r_axis, self.z_axis, 
                                  ne_data.T,
                                  cmap=self.cmap_density,
                                  norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        axes[0,0].set_title('Electron Density')
        axes[0,0].set_xlabel('r (m)')
        axes[0,0].set_ylabel('z (m)')
        plt.colorbar(im0, ax=axes[0,0], label='ne (m^-3)')
        
        # 이온 밀도
        if ni is not None:
            ni_data = np.abs(self._reshape_to_2d(ni))
            vmin, vmax = get_vmin_vmax(ni_data)
            im1 = axes[0,1].pcolormesh(self.r_axis, self.z_axis,
                                      ni_data.T,
                                      cmap=self.cmap_density,
                                      norm=colors.LogNorm(vmin=vmin, vmax=vmax))
            axes[0,1].set_title('Ion Density')
            axes[0,1].set_xlabel('r (m)')
            axes[0,1].set_ylabel('z (m)')
            plt.colorbar(im1, ax=axes[0,1], label='ni (m^-3)')
        
        # 중성 가스 밀도
        if ng is not None:
            ng_data = np.abs(self._reshape_to_2d(ng))
            vmin, vmax = get_vmin_vmax(ng_data)
            im2 = axes[1,0].pcolormesh(self.r_axis, self.z_axis,
                                      ng_data.T,
                                      cmap=self.cmap_density,
                                      norm=colors.LogNorm(vmin=vmin, vmax=vmax))
            axes[1,0].set_title('Neutral Gas Density')
            axes[1,0].set_xlabel('r (m)')
            axes[1,0].set_ylabel('z (m)')
            plt.colorbar(im2, ax=axes[1,0], label='ng (m^-3)')
        
        # 준안정 상태 밀도
        if nms is not None:
            nms_data = np.abs(self._reshape_to_2d(nms))
            vmin, vmax = get_vmin_vmax(nms_data)
            im3 = axes[1,1].pcolormesh(self.r_axis, self.z_axis,
                                      nms_data.T,
                                      cmap=self.cmap_density,
                                      norm=colors.LogNorm(vmin=vmin, vmax=vmax))
            axes[1,1].set_title('Metastable Density')
            axes[1,1].set_xlabel('r (m)')
            axes[1,1].set_ylabel('z (m)')
            plt.colorbar(im3, ax=axes[1,1], label='nms (m^-3)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_temperature_distribution(self,
                                    Te: CellVariable,
                                    title: str = "Electron Temperature Distribution",
                                    save_path: Optional[str] = None) -> None:
        """
        전자 온도 분포를 2D 컬러맵으로 표시

        Args:
            Te: 전자 온도
            title: 그래프 제목
            save_path: 저장할 파일 경로 (선택)
        """
        plt.figure(figsize=(10, 8))
        
        im = plt.pcolormesh(self.r_axis, self.z_axis,
                           np.abs(self._reshape_to_2d(Te)).T,
                           cmap=self.cmap_temperature)
        
        plt.title(title)
        plt.xlabel('r (m)')
        plt.ylabel('z (m)')
        plt.colorbar(im, label='Te (eV)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_electric_field_distribution(self,
                                       E_field: CellVariable,
                                       title: str = "Electric Field Distribution",
                                       save_path: Optional[str] = None) -> None:
        """
        전기장 분포를 2D 컬러맵으로 표시

        Args:
            E_field: 전기장
            title: 그래프 제목
            save_path: 저장할 파일 경로 (선택)
        """
        plt.figure(figsize=(10, 8))
        
        im = plt.pcolormesh(self.r_axis, self.z_axis,
                           np.abs(self._reshape_to_2d(E_field)).T,
                           cmap=self.cmap_field)
        
        plt.title(title)
        plt.xlabel('r (m)')
        plt.ylabel('z (m)')
        plt.colorbar(im, label='E (V/m)')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_radial_profiles(self,
                           ne: CellVariable,
                           Te: CellVariable,
                           E_field: CellVariable,
                           z_pos: float,
                           title: str = "Radial Profiles",
                           save_path: Optional[str] = None) -> None:
        """
        특정 z 위치에서의 반경 방향 프로파일을 표시

        Args:
            ne: 전자 밀도
            Te: 전자 온도
            E_field: 전기장
            z_pos: z 위치 (m)
            title: 그래프 제목
            save_path: 저장할 파일 경로 (선택)
        """
        # z_pos에 가장 가까운 인덱스 찾기
        z_idx = np.abs(self.z_axis - z_pos).argmin()
        
        plt.figure(figsize=(12, 8))
        
        # 전자 밀도
        plt.subplot(3, 1, 1)
        plt.semilogy(self.r_axis, np.abs(self._reshape_to_2d(ne))[:, z_idx], 'b-', label='ne')
        plt.title(f'{title} at z = {z_pos:.3f}m')
        plt.ylabel('Density (m^-3)')
        plt.legend()
        plt.grid(True)
        
        # 전자 온도
        plt.subplot(3, 1, 2)
        plt.plot(self.r_axis, np.abs(self._reshape_to_2d(Te))[:, z_idx], 'r-', label='Te')
        plt.ylabel('Temperature (eV)')
        plt.legend()
        plt.grid(True)
        
        # 전기장
        plt.subplot(3, 1, 3)
        plt.plot(self.r_axis, np.abs(self._reshape_to_2d(E_field))[:, z_idx], 'g-', label='E')
        plt.xlabel('r (m)')
        plt.ylabel('Electric Field (V/m)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_axial_profiles(self,
                          ne: CellVariable,
                          Te: CellVariable,
                          E_field: CellVariable,
                          r_pos: float,
                          title: str = "Axial Profiles",
                          save_path: Optional[str] = None) -> None:
        """
        특정 r 위치에서의 축 방향 프로파일을 표시

        Args:
            ne: 전자 밀도
            Te: 전자 온도
            E_field: 전기장
            r_pos: r 위치 (m)
            title: 그래프 제목
            save_path: 저장할 파일 경로 (선택)
        """
        # r_pos에 가장 가까운 인덱스 찾기
        r_idx = np.abs(self.r_axis - r_pos).argmin()
        
        plt.figure(figsize=(12, 8))
        
        # 전자 밀도
        plt.subplot(3, 1, 1)
        plt.semilogy(self.z_axis, np.abs(self._reshape_to_2d(ne))[r_idx, :], 'b-', label='ne')
        plt.title(f'{title} at r = {r_pos:.3f}m')
        plt.ylabel('Density (m^-3)')
        plt.legend()
        plt.grid(True)
        
        # 전자 온도
        plt.subplot(3, 1, 2)
        plt.plot(self.z_axis, np.abs(self._reshape_to_2d(Te))[r_idx, :], 'r-', label='Te')
        plt.ylabel('Temperature (eV)')
        plt.legend()
        plt.grid(True)
        
        # 전기장
        plt.subplot(3, 1, 3)
        plt.plot(self.z_axis, np.abs(self._reshape_to_2d(E_field))[r_idx, :], 'g-', label='E')
        plt.xlabel('z (m)')
        plt.ylabel('Electric Field (V/m)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_time_evolution(self,
                          time_history: Dict[str, List[float]],
                          title: str = "Time Evolution of Plasma Parameters",
                          save_path: Optional[str] = None) -> None:
        """
        시간에 따른 플라즈마 파라미터 변화를 그래프로 표시

        Args:
            time_history: 시간 이력 데이터
            title: 그래프 제목
            save_path: 저장할 파일 경로 (선택)
        """
        # 시간 배열 생성 (time_history의 길이 사용)
        n_steps = len(time_history['ne_avg'])
        time_points = np.linspace(0, self.current_time, n_steps)
        
        plt.figure(figsize=(12, 8))
        
        # 밀도 변화
        plt.subplot(2, 2, 1)
        plt.semilogy(time_points, time_history['ne_avg'], 'b-', label='ne')
        plt.semilogy(time_points, time_history['ni_avg'], 'r--', label='ni')
        plt.title('Density Evolution')
        plt.xlabel('Time (s)')
        plt.ylabel('Density (m^-3)')
        plt.legend()
        plt.grid(True)
        
        # 온도 변화
        plt.subplot(2, 2, 2)
        plt.plot(time_points, time_history['Te_avg'], 'g-', label='Te')
        plt.title('Temperature Evolution')
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (eV)')
        plt.legend()
        plt.grid(True)
        
        # 전기장 변화
        plt.subplot(2, 2, 3)
        plt.semilogy(time_points, time_history['E_avg'], 'm-', label='E')
        plt.title('Electric Field Evolution')
        plt.xlabel('Time (s)')
        plt.ylabel('Electric Field (V/m)')
        plt.legend()
        plt.grid(True)
        
        # 플라즈마 저항 변화
        plt.subplot(2, 2, 4)
        plt.semilogy(time_points, time_history['Rp'], 'k-', label='Rp')
        plt.title('Plasma Resistance Evolution')
        plt.xlabel('Time (s)')
        plt.ylabel('Resistance (Ohm)')
        plt.legend()
        plt.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

    def plot_resistance_comparison(self, save_path: Optional[str] = None) -> None:
        """회로 모델과 실제 플라즈마 저항의 비교 그래프"""
        import matplotlib.pyplot as plt
        
        time_points = np.arange(0, self.current_time + self.time_step, self.time_step)
        
        plt.figure(figsize=(15, 10))
        
        # 저항 비교
        plt.subplot(2, 2, 1)
        plt.plot(time_points, self.resistance_history['circuit_Rp'], 
                label='Circuit Model Rp', linestyle='-')
        plt.plot(time_points, self.resistance_history['actual_Rp'], 
                label='Actual Rp', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Resistance (Ohm)')
        plt.legend()
        plt.title('Plasma Resistance Comparison')
        
        # 저항 비율
        plt.subplot(2, 2, 2)
        ratio = np.array(self.resistance_history['circuit_Rp']) / \
                (np.array(self.resistance_history['actual_Rp']) + 1e-10)
        plt.plot(time_points, ratio)
        plt.xlabel('Time (s)')
        plt.ylabel('Circuit Rp / Actual Rp')
        plt.title('Resistance Ratio')
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        
        # 전력 비교
        plt.subplot(2, 2, 3)
        plt.plot(time_points, self.resistance_history['input_power'], 
                label='Input Power', linestyle='-')
        plt.plot(time_points, self.resistance_history['power_dissipation'], 
                label='Dissipated Power', linestyle='--')
        plt.xlabel('Time (s)')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.title('Power Comparison')
        
        # 효율
        plt.subplot(2, 2, 4)
        plt.plot(time_points, np.array(self.resistance_history['efficiency'])*100)
        plt.xlabel('Time (s)')
        plt.ylabel('Efficiency (%)')
        plt.title('Power Efficiency')
        plt.axhline(y=100, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot_energy_conservation(self, time_history: Dict[str, List[float]], 
                               title: str = "Energy Conservation in Plasma",
                               save_path: Optional[str] = None) -> None:
        """에너지 보존 시각화"""
        plt.figure(figsize=(12, 6))
        
        # 시간 축
        time = np.array(time_history['time'])
        
        # 에너지 입력 (가열)
        plt.plot(time, time_history['P_ohm'], label='Ohmic Heating', color='red')
        plt.plot(time, time_history['P_stoch'], label='Stochastic Heating', color='orange')
        plt.plot(time, time_history['P_res'], label='Resonant Heating', color='yellow')
        
        # 에너지 출력 (냉각)
        plt.plot(time, time_history['P_coll'], label='Collisional Cooling', color='blue')
        plt.plot(time, time_history['P_ion'], label='Ionization Loss', color='green')
        plt.plot(time, time_history['P_ex'], label='Excitation Loss', color='purple')
        plt.plot(time, time_history['P_wall'], label='Wall Loss', color='brown')
        
        # 순 에너지 변화
        plt.plot(time, time_history['P_net'], label='Net Power', color='black', linewidth=2)
        
        plt.xlabel('Time [s]')
        plt.ylabel('Power [W]')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_entropy_generation(self, time_history: Dict[str, List[float]],
                              title: str = "Entropy Generation in Plasma",
                              save_path: Optional[str] = None) -> None:
        """엔트로피 생성 시각화"""
        plt.figure(figsize=(12, 6))
        
        # 시간 축
        time = np.array(time_history['time'])
        
        # 엔트로피 생성 항
        plt.plot(time, time_history['S_coll'], label='Collisional Entropy', color='blue')
        plt.plot(time, time_history['S_ion'], label='Ionization Entropy', color='red')
        plt.plot(time, time_history['S_ex'], label='Excitation Entropy', color='green')
        plt.plot(time, time_history['S_rec'], label='Recombination Entropy', color='purple')
        
        # 총 엔트로피 생성
        plt.plot(time, time_history['S_total'], label='Total Entropy', color='black', linewidth=2)
        
        plt.xlabel('Time [s]')
        plt.ylabel('Entropy Generation Rate [J/K/s]')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
