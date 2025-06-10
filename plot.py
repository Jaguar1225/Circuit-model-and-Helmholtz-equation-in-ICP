import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Dict, Tuple
from matplotlib.colors import LogNorm
import matplotlib.font_manager as fm

# 한글 폰트 설정 강화
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# 컬러맵 설정
CMAP_DICT = {
    'temperature': 'hot',
    'density': 'viridis',
    'field': 'coolwarm',
    'conductivity': 'plasma',
    'potential': 'magma'
}

def safe_log_vmin(arr):
    arr_pos = arr[arr > 0]
    if arr_pos.size == 0:
        return 1e-10
    return arr_pos.min()

def safe_log_norm(arr):
    arr_pos = arr[arr > 0]
    vmax = arr.max() if arr.size > 0 else 1.0
    if arr_pos.size == 0 or vmax <= 0:
        return None, vmax
    vmin = arr_pos.min()
    if vmin >= vmax:
        return None, vmax
    return LogNorm(vmin=vmin, vmax=vmax), vmax

def safe_pcolormesh(ax, R, Z, data, **kwargs):
    # nan/inf를 0으로 대체
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return ax.pcolormesh(R, Z, data, **kwargs)

def setup_plot_style():
    """공통 플롯 스타일 설정"""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 10)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.2

def plot_coil_location(ax, config):
    """코일 위치를 정확하게 표시하는 함수"""
    coil_r = config['coil']['coil_radius_m'] * 100  # cm 단위로 변환
    coil_z = 0.0
    wire_radius = config['coil']['wire_radius_m'] * 100  # cm 단위로 변환
    
    # 코일 중심선 표시
    theta = np.linspace(0, 2*np.pi, 100)
    x = coil_r * np.cos(theta)
    y = coil_r * np.sin(theta)
    ax.plot(x, y, 'r--', linewidth=1, alpha=0.5, label='Coil Center')
    
    # 코일 와이어 단면 표시 (상단과 하단)
    wire_top = plt.Circle((coil_r, wire_radius), wire_radius, 
                         fill=False, color='red', linestyle='-', linewidth=1.5)
    wire_bottom = plt.Circle((coil_r, -wire_radius), wire_radius,
                            fill=False, color='red', linestyle='-', linewidth=1.5)
    ax.add_patch(wire_top)
    ax.add_patch(wire_bottom)
    
    # 코일 정보 텍스트
    info_text = (
        f"Coil Radius: {coil_r:.1f} cm\n"
        f"Wire Radius: {wire_radius:.1f} cm"
    )
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)

def plot_field_distribution(result: Dict, save_path: str = 'results/field_distribution.png'):
    """전자기장 및 플라즈마 분포 시각화"""
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Plasma Physical Quantities Distribution', fontsize=20, y=1.05)
    
    # 1. 전자 온도 분포
    r = result['mesh']['r'] * 100  # cm 단위로 변환
    z = result['mesh']['z'] * 100
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    im1 = safe_pcolormesh(axes[0], R, Z, np.abs(result['Te']),
                         shading='auto', cmap=CMAP_DICT['temperature'])
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Electron Temperature [eV]', fontsize=12)
    axes[0].set_xlabel('Radius [cm]', fontsize=14)
    axes[0].set_ylabel('Height [cm]', fontsize=14)
    axes[0].set_title('Electron Temperature Distribution', fontsize=16)
    axes[0].grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    axes[0].set_aspect('equal')
    plot_coil_location(axes[0], result['config'])
    
    # 2. 전기장 크기
    E_mag = np.sqrt(np.abs(result['E_field'][0])**2 + 
                   np.abs(result['E_field'][1])**2 + 
                   np.abs(result['E_field'][2])**2)
    im2 = safe_pcolormesh(axes[1], R, Z, E_mag,
                         shading='auto', cmap=CMAP_DICT['field'])
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Electric Field Magnitude [V/m]', fontsize=12)
    axes[1].set_xlabel('Radius [cm]', fontsize=14)
    axes[1].set_ylabel('Height [cm]', fontsize=14)
    axes[1].set_title('Electric Field Magnitude Distribution', fontsize=16)
    axes[1].grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    axes[1].set_aspect('equal')
    plot_coil_location(axes[1], result['config'])
    
    # 3. 플라즈마 전도도
    im3 = safe_pcolormesh(axes[2], R, Z, np.abs(result['sigma_p']),
                         shading='auto', cmap=CMAP_DICT['conductivity'])
    cbar3 = plt.colorbar(im3, ax=axes[2])
    cbar3.set_label('Plasma Conductivity [S/m]', fontsize=12)
    axes[2].set_xlabel('Radius [cm]', fontsize=14)
    axes[2].set_ylabel('Height [cm]', fontsize=14)
    axes[2].set_title('Plasma Conductivity Distribution', fontsize=16)
    axes[2].grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    axes[2].set_aspect('equal')
    plot_coil_location(axes[2], result['config'])
    
    # 모든 서브플롯에 대해 동일한 축 범위 설정
    x_min, x_max = r.min(), r.max()
    z_min, z_max = z.min(), z.max()
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_circuit_parameters(result: Dict, save_dir: Path) -> None:
    """회로 파라미터 시각화"""
    setup_plot_style()
    
    circuit = result['circuit_params']
    
    # 임피던스 스미스 차트
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle('Impedance Smith Chart', fontsize=20, y=1.05)
    
    # 단위원 그리기
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=2)
    ax.add_artist(circle)
    
    # 임피던스 점 표시
    Z = circuit['total_impedance'] if 'total_impedance' in circuit else circuit.get('Z_total', 0)
    Z_norm = (Z - 50) / (Z + 50) if (Z + 50) != 0 else 0
    ax.plot(np.real(Z_norm), np.imag(Z_norm), 'ro', markersize=10, label='Impedance')
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Real Part', fontsize=14)
    ax.set_ylabel('Imaginary Part', fontsize=14)
    
    # 추가 정보
    info_text = (
        f"Impedance: {abs(Z):.1f} Ω\n"
        f"Phase Angle: {np.angle(Z, deg=True):.1f}°\n"
        f"Current: {circuit.get('rms_current', circuit.get('I_rms', 0)):.1f} A\n"
        f"Voltage: {circuit.get('rms_voltage', circuit.get('V_rms', 0)):.1f} V"
    )
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=12)
    
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / "circuit_parameters.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_vector_potential(result, config, save_dir):
    """벡터 포텐셜 분포 시각화"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle('Vector Potential Distribution', fontsize=20, y=1.05)
    
    A = np.abs(result['A_theta'])
    r = result['mesh']['r'] * 100
    z = result['mesh']['z'] * 100
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    norm, vmax = safe_log_norm(A)
    if np.all(A == 0):
        print('[경고] 모든 벡터 포텐셜 데이터가 0입니다. 플롯을 건너뜁니다.')
        plt.close()
        return
        
    im = safe_pcolormesh(ax, R, Z, A,
        shading='auto',
        cmap=CMAP_DICT['potential'],
        norm=norm if norm else None
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Vector Potential [Wb/m]', fontsize=12)
    
    ax.set_xlabel('Radius [cm]', fontsize=14)
    ax.set_ylabel('Height [cm]', fontsize=14)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')
    
    plot_coil_location(ax, config)
    
    # 축 범위 설정
    ax.set_xlim(r.min(), r.max())
    ax.set_ylim(z.min(), z.max())
    
    plt.tight_layout()
    plt.savefig(save_dir / 'vector_potential.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_electric_field(result, config, save_dir):
    """전기장 분포 시각화"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle('Electric Field Distribution', fontsize=20, y=1.05)
    
    Er, Ez, E_theta = result['E_field']
    E = np.sqrt(np.abs(Er)**2 + np.abs(Ez)**2 + np.abs(E_theta)**2)
    r = result['mesh']['r'] * 100
    z = result['mesh']['z'] * 100
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    norm, vmax = safe_log_norm(E)
    if np.all(E == 0):
        print('[경고] 모든 전기장 데이터가 0입니다. 플롯을 건너뜁니다.')
        plt.close()
        return
        
    im = safe_pcolormesh(ax, R, Z, E,
        shading='auto',
        cmap=CMAP_DICT['field'],
        norm=norm if norm else None
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Electric Field Magnitude [V/m]', fontsize=12)
    
    ax.set_xlabel('Radius [cm]', fontsize=14)
    ax.set_ylabel('Height [cm]', fontsize=14)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')
    
    plot_coil_location(ax, config)
    
    # 축 범위 설정
    ax.set_xlim(r.min(), r.max())
    ax.set_ylim(z.min(), z.max())
    
    plt.tight_layout()
    plt.savefig(save_dir / 'electric_field.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_magnetic_field(result, config, save_dir):
    """자기장 분포 시각화"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle('Magnetic Field Distribution', fontsize=20, y=1.05)
    
    B = np.abs(result['B_field'])
    r = result['mesh']['r'] * 100
    z = result['mesh']['z'] * 100
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    norm, vmax = safe_log_norm(B)
    if np.all(B == 0):
        print('[경고] 모든 자기장 데이터가 0입니다. 플롯을 건너뜁니다.')
        plt.close()
        return
        
    im = safe_pcolormesh(ax, R, Z, B,
        shading='auto',
        cmap=CMAP_DICT['field'],
        norm=norm if norm else None
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnetic Field Magnitude [T]', fontsize=12)
    
    ax.set_xlabel('Radius [cm]', fontsize=14)
    ax.set_ylabel('Height [cm]', fontsize=14)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')
    
    plot_coil_location(ax, config)
    
    # 축 범위 설정
    ax.set_xlim(r.min(), r.max())
    ax.set_ylim(z.min(), z.max())
    
    plt.tight_layout()
    plt.savefig(save_dir / 'magnetic_field.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_electron_temperature(result, config, save_dir):
    """전자 온도 분포 시각화"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle('Electron Temperature Distribution', fontsize=20, y=1.05)
    
    T_e = np.abs(result['Te'])
    r = result['mesh']['r'] * 100
    z = result['mesh']['z'] * 100
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    norm, vmax = safe_log_norm(T_e)
    if np.all(T_e == 0):
        print('[경고] 모든 전자 온도 데이터가 0입니다. 플롯을 건너뜁니다.')
        plt.close()
        return
        
    im = safe_pcolormesh(ax, R, Z, T_e,
        shading='auto',
        cmap=CMAP_DICT['temperature'],
        norm=norm if norm else None
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Electron Temperature [eV]', fontsize=12)
    
    ax.set_xlabel('Radius [cm]', fontsize=14)
    ax.set_ylabel('Height [cm]', fontsize=14)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')
    
    plot_coil_location(ax, config)
    
    # 축 범위 설정
    ax.set_xlim(r.min(), r.max())
    ax.set_ylim(z.min(), z.max())
    
    plt.tight_layout()
    plt.savefig(save_dir / 'electron_temperature.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_electron_density(result, config, save_dir):
    """전자 밀도 분포 시각화"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle('Electron Density Distribution', fontsize=20, y=1.05)
    
    sigma_p = np.abs(result['sigma_p'])
    T_e = np.abs(result['Te'])
    nu = np.abs(result['nu'])
    e = config['constants']['e']
    m_e = config['constants']['m_e']
    n_e = sigma_p * m_e * nu / (e**2)
    
    r = result['mesh']['r'] * 100
    z = result['mesh']['z'] * 100
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    norm, vmax = safe_log_norm(n_e)
    if np.all(n_e == 0):
        print('[경고] 모든 전자 밀도 데이터가 0입니다. 플롯을 건너뜁니다.')
        plt.close()
        return
        
    im = safe_pcolormesh(ax, R, Z, n_e,
        shading='auto',
        cmap=CMAP_DICT['density'],
        norm=norm if norm else None
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Electron Density [m$^{-3}$]', fontsize=12)
    
    ax.set_xlabel('Radius [cm]', fontsize=14)
    ax.set_ylabel('Height [cm]', fontsize=14)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')
    
    plot_coil_location(ax, config)
    
    # 축 범위 설정
    ax.set_xlim(r.min(), r.max())
    ax.set_ylim(z.min(), z.max())
    
    plt.tight_layout()
    plt.savefig(save_dir / 'electron_density.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_conductivity(result, config, save_dir):
    """플라즈마 전도도 분포 시각화"""
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle('Plasma Conductivity Distribution', fontsize=20, y=1.05)
    
    sigma_p = np.abs(result['sigma_p'])
    r = result['mesh']['r'] * 100
    z = result['mesh']['z'] * 100
    R, Z = np.meshgrid(r, z, indexing='ij')
    
    norm, vmax = safe_log_norm(sigma_p)
    if np.all(sigma_p == 0):
        print('[경고] 모든 전도도 데이터가 0입니다. 플롯을 건너뜁니다.')
        plt.close()
        return
        
    im = safe_pcolormesh(ax, R, Z, sigma_p,
        shading='auto',
        cmap=CMAP_DICT['conductivity'],
        norm=norm if norm else None
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Conductivity [S/m]', fontsize=12)
    
    ax.set_xlabel('Radius [cm]', fontsize=14)
    ax.set_ylabel('Height [cm]', fontsize=14)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')
    
    plot_coil_location(ax, config)
    
    # 축 범위 설정
    ax.set_xlim(r.min(), r.max())
    ax.set_ylim(z.min(), z.max())
    
    plt.tight_layout()
    plt.savefig(save_dir / 'conductivity.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_results(result: Dict, config: Dict, save_dir: Path) -> None:
    """
    모든 계산 결과를 시각화하고 저장합니다.
    
    Args:
        result: 계산 결과 딕셔너리
        config: 설정 딕셔너리
        save_dir: 결과 저장 디렉토리 (Path 객체)
    """
    # 그래프 스타일 설정
    plt.style.use('seaborn-v0_8')
    
    # 벡터 포텐셜 분포
    plot_vector_potential(result, config, save_dir)
    
    # 전기장 분포
    plot_electric_field(result, config, save_dir)
    
    # 자기장 분포
    plot_magnetic_field(result, config, save_dir)
    
    # 전자 온도 분포
    plot_electron_temperature(result, config, save_dir)
    
    # 전자 밀도 분포
    plot_electron_density(result, config, save_dir)
    
    # 전도도 분포
    plot_conductivity(result, config, save_dir)
    
    # 회로 파라미터
    plot_circuit_parameters(result, save_dir)
    
    # 새로운 플롯 추가
    plot_time_evolution(result, save_dir)
    plot_reaction_rates(result, save_dir)
    
    plt.close('all')

def setup_plot(title: str, xlabel: str, ylabel: str, save_path: Path) -> Tuple[plt.Figure, plt.Axes]:
    """
    공통 플롯 설정을 수행합니다.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    return fig, ax

def save_plot(fig: plt.Figure, save_path: Path, dpi: int = 300) -> None:
    """
    플롯을 저장하고 메모리를 정리합니다.
    """
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def plot_time_evolution(result: Dict, save_dir: Path) -> None:
    """시간에 따른 물리량 변화 시각화"""
    # 구현 필요

def plot_reaction_rates(result: Dict, save_dir: Path) -> None:
    """반응 속도 분포 시각화"""
    # 구현 필요
