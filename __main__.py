import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from cal import create_mesh, SelfConsistentSolver
from plot import plot_all_results
from typing import Dict

def load_config() -> Dict:
    """
    config 폴더에서 설정 파일들을 불러와서 하나의 설정 딕셔너리로 병합합니다.
    각 설정 파일의 키는 파일 이름에서 .json을 제외한 이름을 사용합니다.
    """
    
    config_dir = 'config'
    config_files = {
        'chamber': 'chamber.json',      # 챔버 구조 및 재질 정보
        'coil': 'coil.json',           # 코일 특성 정보
        'process': 'process.json',     # 공정 조건 정보
        'initial': 'initial.json',     # 초기 조건 정보
        'mesh': 'mesh.json',           # 격자 설정 정보
        'constants': 'constants.json',  # 물리 상수 정보
        'chemistry': 'chemistry.json'  # 화학 반응 정보
    }

    config = {}
    # 각 설정 파일 로드
    for key, file_name in config_files.items():
        file_path = os.path.join(config_dir, file_name)
        try:
            with open(file_path, 'r') as f:
                file_config = json.load(f)
                config[key] = file_config
        except FileNotFoundError:
            print(f"경고: {file_path} 파일을 찾을 수 없습니다.")
            raise
        except json.JSONDecodeError:
            print(f"경고: {file_path} 파일의 JSON 형식이 올바르지 않습니다.")
            raise
    return config

def validate_config(config: Dict) -> None:
    """
    설정값들의 유효성을 검사합니다.
    """
    # 필수 설정 파일 존재 확인
    required_keys = ['chamber', 'coil', 'process', 'initial', 'mesh', 'constants','chemistry']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"필수 설정 '{key}'가 없습니다.")
    
    # 챔버 설정 검증
    chamber = config['chamber']
    if chamber['chamber_height_m'] <= 0 or chamber['chamber_radius_m'] <= 0:
        raise ValueError("챔버 크기는 양수여야 합니다.")
    
    # 코일 설정 검증
    coil = config['coil']
    if coil['coil_radius_m'] >= chamber['chamber_radius_m']:
        raise ValueError("코일 반지름은 챔버 반지름보다 작아야 합니다.")
    
    # 공정 설정 검증
    process = config['process']
    if process['input_power_W'] <= 0:
        raise ValueError("입력 전력은 양수여야 합니다.")
    
    # 격자 설정 검증
    mesh = config['mesh']
    if mesh['radial_grid_points'] < 10 or mesh['axial_grid_points'] < 10:
        raise ValueError("격자 수는 최소 10개 이상이어야 합니다.")
    
    # 화학 반응 설정 검증
    if 'chemistry' not in config:
        raise ValueError("화학 반응 설정이 없습니다.")
    
    # 시간 의존적 해석 설정 검증
    if 'process' in config:
        if 'time_step_s' in config['process']:
            if config['process']['time_step_s'] <= 0:
                raise ValueError("시간 스텝은 양수여야 합니다.")
        if 'simulation_time_s' in config['process']:
            if config['process']['simulation_time_s'] <= 0:
                raise ValueError("시뮬레이션 시간은 양수여야 합니다.")

def create_result_directory() -> Path:
    """
    결과를 저장할 디렉토리를 생성합니다.
    형식: results/YYYYMMDD_HHMMSS/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path('results') / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 하위 디렉토리 생성
    (result_dir / 'plots').mkdir(exist_ok=True)
    (result_dir / 'data').mkdir(exist_ok=True)
    
    return result_dir

def save_results(result: Dict, config: Dict, result_dir: Path) -> None:
    """
    계산 결과를 체계적으로 저장합니다.
    
    저장 구조:
    results/YYYYMMDD_HHMMSS/
    ├── plots/
    │   ├── vector_potential.png
    │   ├── electric_field.png
    │   ├── magnetic_field.png
    │   ├── electron_temperature.png
    │   ├── electron_density.png
    │   ├── conductivity.png
    │   └── circuit_parameters.png
    ├── data/
    │   ├── calculation_result.npz
    │   ├── config.json
    │   └── summary.txt
    └── README.md
    """
    # 계산 결과 저장
    np.savez(result_dir / 'data' / 'calculation_result.npz',
             A_theta=result['A_theta'],
             E_field=result['E_field'],
             B_field=result['B_field'],
             Te=result['Te'],
             nu=result['nu'],
             sigma_p=result['sigma_p'],
             Rp=result['Rp'],
             circuit_params=result['circuit_params'],
             mesh_r=result['mesh']['r'],
             mesh_z=result['mesh']['z'])
    
    # 설정 정보 저장
    with open(result_dir / 'data' / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # 요약 정보 저장
    save_summary(result, config, result_dir)
    
    # README 파일 생성
    create_readme(result_dir, config)

def save_summary(result: Dict, config: Dict, result_dir: Path) -> None:
    """
    계산 결과의 요약 정보를 텍스트 파일로 저장합니다.
    """
    summary = []
    summary.append("=== ICP 시뮬레이션 결과 요약 ===")
    summary.append(f"계산 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 공정 조건
    summary.append("--- 공정 조건 ---")
    summary.append(f"RF 주파수: {config['process']['radio_frequency_Hz']/1e6:.2f} MHz")
    summary.append(f"입력 전력: {config['process']['input_power_W']:.1f} W")
    summary.append(f"가스 종류: {config['chamber']['gas_type']}")
    summary.append(f"작동 압력: {config['chamber']['pressure_Pa']:.2e} Pa\n")
    
    # 코일 정보
    summary.append("--- 코일 정보 ---")
    summary.append(f"코일 반지름: {config['coil']['coil_radius_m']*100:.1f} cm")
    summary.append(f"코일 턴 수: {config['coil']['number_of_turns']}")
    summary.append(f"코일 인덕턴스: {config['coil']['coil_inductance_H']*1e6:.2f} μH\n")
    
    # 플라즈마 특성
    summary.append("--- 플라즈마 특성 ---")
    Te_avg = np.mean(result['Te'])
    ne_avg = np.mean(result['sigma_p'] / (result['Te'] * config['constants']['e']))
    summary.append(f"평균 전자 온도: {Te_avg:.2f} eV")
    summary.append(f"평균 전자 밀도: {ne_avg:.2e} m^-3")
    summary.append(f"플라즈마 저항: {result['Rp']:.2f} Ω")
    
    # 회로 특성
    summary.append("\n--- 회로 특성 ---")
    circuit = result['circuit_params']
    summary.append(f"총 임피던스: {abs(circuit['total_impedance']):.2f} Ω")
    summary.append(f"위상각: {np.angle(circuit['total_impedance'], deg=True):.1f}°")
    summary.append(f"RMS 전류: {circuit['rms_current']:.2f} A")
    summary.append(f"RMS 전압: {circuit['rms_voltage']:.2f} V")
    summary.append(f"플라즈마 소비 전력: {circuit['plasma_power']:.1f} W")
    
    # 파일 저장
    with open(result_dir / 'data' / 'summary.txt', 'w') as f:
        f.write('\n'.join(summary))

def create_readme(result_dir: Path, config: Dict) -> None:
    """
    결과 폴더에 README.md 파일을 생성합니다.
    """
    readme = []
    readme.append("# ICP 시뮬레이션 결과")
    readme.append(f"\n계산 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    readme.append("\n## 폴더 구조")
    readme.append("```")
    readme.append("results/YYYYMMDD_HHMMSS/")
    readme.append("├── plots/          # 시각화 결과")
    readme.append("├── data/           # 계산 데이터")
    readme.append("└── README.md       # 이 파일")
    readme.append("```")
    
    readme.append("\n## 시뮬레이션 조건")
    readme.append(f"- RF 주파수: {config['process']['radio_frequency_Hz']/1e6:.2f} MHz")
    readme.append(f"- 입력 전력: {config['process']['input_power_W']:.1f} W")
    readme.append(f"- 가스: {config['chamber']['gas_type']}")
    readme.append(f"- 압력: {config['chamber']['pressure_Pa']:.2e} Pa")
    
    readme.append("\n## 파일 설명")
    readme.append("- `plots/*.png`: 각 물리량의 2D 분포도")
    readme.append("- `data/calculation_result.npz`: 계산 결과 데이터")
    readme.append("- `data/config.json`: 시뮬레이션 설정")
    readme.append("- `data/summary.txt`: 주요 결과 요약")
    
    # 파일 저장
    with open(result_dir / 'README.md', 'w') as f:
        f.write('\n'.join(readme))

def main():
    try:
        # 결과 저장 디렉토리 생성
        result_dir = create_result_directory()
        print(f"결과 저장 경로: {result_dir}")
        
        # 설정 파일 로드
        print("설정 파일 로드 중...")
        config = load_config()
        
        # 설정 유효성 검사
        print("설정 유효성 검사 중...")
        validate_config(config)
        
        # 격자 생성
        print("격자 생성 중...")
        mesh = create_mesh(config['mesh'])
        
        # 자기 일관성 해 계산
        print("자기 일관성 해 계산 시작...")
        solver = SelfConsistentSolver(mesh, config)
        result = solver.solve(
            max_iter=100,
            tolerance=config['initial']['convergence_tolerance']
        )
        
        # 결과 시각화 및 저장
        print("결과 시각화 및 저장 중...")
        plot_all_results(result, config, save_dir=result_dir / 'plots')
        
        # 결과 저장
        print("결과 데이터 저장 중...")
        save_results(result, config, result_dir)
        
        print("\n계산이 완료되었습니다!")
        print(f"결과는 '{result_dir}' 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        raise

if __name__ == '__main__':
    main()