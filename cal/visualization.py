import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class Visualization:
    def __init__(self, config: dict, results_dir: str):
        """
        시각화 클래스 초기화
        
        Args:
            config: 설정 딕셔너리
            results_dir: 결과 저장 디렉토리
        """
        self.config = config
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)

    def save_results(
            self,
            step: int,
            ne: np.ndarray,
            ni: np.ndarray,
            ng: np.ndarray,
            nms: np.ndarray,
            Te: np.ndarray,
            sigma_p: np.ndarray,
            E_field: np.ndarray,
            Rp: float
            ) -> None:
        """
        계산 결과 저장 및 시각화
        
        Args:
            step: 현재 스텝
            ne, ni, ng, nms: 밀도 배열
            Te: 전자 온도 배열
            sigma_p: 플라즈마 전도도 배열
            E_field: 전기장 배열
            Rp: 플라즈마 저항
        """
        # 시간 계산
        t = step * self.config['simulation']['dt']
        
        # 복소수 배열의 절대값 사용
        ne_abs = np.abs(ne)
        ni_abs = np.abs(ni)
        nms_abs = np.abs(nms)
        sigma_p_abs = np.abs(sigma_p)
        E_field_abs = np.abs(E_field)

        # 밀도 분포 시각화
        self._plot_density(step, t, ne_abs, ni_abs, ng, nms_abs)
        
        # 온도 분포 시각화
        self._plot_temperature(step, t, Te)
        
        # 전기장 분포 시각화
        self._plot_electric_field(step, t, E_field_abs)
        
        # 전도도 분포 시각화
        self._plot_conductivity(step, t, sigma_p_abs)
        
        # 데이터 저장
        self._save_data(step, t, ne, ni, ng, nms, Te, sigma_p, E_field, Rp)

    def _plot_density(
            self,
            step: int,
            t: float,
            ne: np.ndarray,
            ni: np.ndarray,
            ng: np.ndarray,
            nms: np.ndarray
            ) -> None:
        """
        밀도 분포 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Density Distribution (t = {t:.2e}s)')

        # 전자 밀도
        im0 = axes[0, 0].imshow(ne, origin='lower', aspect='auto')
        axes[0, 0].set_title('Electron Density')
        plt.colorbar(im0, ax=axes[0, 0], label='m^-3')

        # 이온 밀도
        im1 = axes[0, 1].imshow(ni, origin='lower', aspect='auto')
        axes[0, 1].set_title('Ion Density')
        plt.colorbar(im1, ax=axes[0, 1], label='m^-3')

        # 중성 가스 밀도
        im2 = axes[1, 0].imshow(np.abs(ng), origin='lower', aspect='auto')
        axes[1, 0].set_title('Neutral Gas Density')
        plt.colorbar(im2, ax=axes[1, 0], label='m^-3')

        # 준안정 상태 밀도
        im3 = axes[1, 1].imshow(nms, origin='lower', aspect='auto')
        axes[1, 1].set_title('Metastable Density')
        plt.colorbar(im3, ax=axes[1, 1], label='m^-3')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'density_t{step:04d}.png'))
        plt.close()

    def _plot_temperature(self, step: int, t: float, Te: np.ndarray) -> None:
        """
        온도 분포 시각화
        """
        plt.figure(figsize=(8, 6))
        plt.title(f'Electron Temperature Distribution (t = {t:.2e}s)')
        im = plt.imshow(np.abs(Te), origin='lower', aspect='auto')
        plt.colorbar(im, label='eV')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'temperature_t{step:04d}.png'))
        plt.close()

    def _plot_electric_field(self, step: int, t: float, E_field: np.ndarray) -> None:
        """
        전기장 분포 시각화
        """
        plt.figure(figsize=(8, 6))
        plt.title(f'Electric Field Distribution (t = {t:.2e}s)')
        im = plt.imshow(E_field, origin='lower', aspect='auto')
        plt.colorbar(im, label='V/m')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'field_t{step:04d}.png'))
        plt.close()

    def _plot_conductivity(self, step: int, t: float, sigma_p: np.ndarray) -> None:
        """
        전도도 분포 시각화
        """
        plt.figure(figsize=(8, 6))
        plt.title(f'Plasma Conductivity Distribution (t = {t:.2e}s)')
        im = plt.imshow(sigma_p, origin='lower', aspect='auto')
        plt.colorbar(im, label='S/m')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'conductivity_t{step:04d}.png'))
        plt.close()

    def _save_data(
            self,
            step: int,
            t: float,
            ne: np.ndarray,
            ni: np.ndarray,
            ng: np.ndarray,
            nms: np.ndarray,
            Te: np.ndarray,
            sigma_p: np.ndarray,
            E_field: np.ndarray,
            Rp: float
            ) -> None:
        """
        계산 데이터 저장
        """
        data = {
            'step': step,
            'time': t,
            'ne': ne,
            'ni': ni,
            'ng': ng,
            'nms': nms,
            'Te': Te,
            'sigma_p': sigma_p,
            'E_field': E_field,
            'Rp': Rp
        }
        
        np.savez(
            os.path.join(self.results_dir, f'data_t{step:04d}.npz'),
            **data
        ) 