import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from cal.mesh import Mesh
from cal.plasma import PlasmaModel
from cal.helmholtz import HelmholtzModel
from cal.circuit import CircuitModel
from cal.visualization import Visualization
from cal.opt import SelfConsistentSolver
from cal.em import EMModel

def load_config() -> dict:
    """
    config 폴더에서 설정 파일들을 불러와서 하나의 설정 딕셔너리로 병합합니다.
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
            with open(file_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config[key] = file_config
        except FileNotFoundError:
            print(f"경고: {file_path} 파일을 찾을 수 없습니다.")
            raise
        except json.JSONDecodeError:
            print(f"경고: {file_path} 파일의 JSON 형식이 올바르지 않습니다.")
            raise

    # simulation 설정 추가 (process.json에서 가져옴)
    config['simulation'] = {
        'total_steps': int(config['process']['simulation_time_s'] / config['process']['time_step_s']),
        'dt': config['process']['time_step_s'],
        'save_interval': 100,  # 100 스텝마다 저장
        'print_interval': 100  # 100 스텝마다 출력
    }

    # 디버깅을 위한 출력 추가
    print(f"\n시뮬레이션 설정:")
    print(f"  총 시뮬레이션 시간: {config['process']['simulation_time_s']:.3e} 초")
    print(f"  시간 스텝: {config['process']['time_step_s']:.3e} 초")
    print(f"  총 스텝 수: {config['simulation']['total_steps']} 스텝")
    print(f"  저장 간격: {config['simulation']['save_interval']} 스텝")
    print(f"  출력 간격: {config['simulation']['print_interval']} 스텝\n")

    # circuit 설정 추가 (coil.json에서 가져옴)
    config['circuit'] = {
        'coil_radius': config['coil']['coil_radius_max'],
        'coil_height': config['coil']['coil_width'],
        'coil_turns': config['coil']['number_of_turns'],
        'radio_frequency_Hz': config['process']['radio_frequency_Hz']
    }

    return config

def validate_config(config: dict) -> bool:
    """
    설정 유효성 검사
    """
    # 필수 키 존재 여부 확인
    required_keys = [
        'chamber', 'mesh', 'simulation', 'chemistry', 'circuit', 'constants'
    ]
    for key in required_keys:
        if key not in config:
            print(f"설정 파일에 필수 키 '{key}'가 없습니다.")
            return False

    # chamber 설정 검사
    chamber_keys = ['chamber_radius_m', 'chamber_height_m']
    for key in chamber_keys:
        if key not in config['chamber']:
            print(f"chamber 설정에 필수 키 '{key}'가 없습니다.")
            return False

    # mesh 설정 검사
    mesh_keys = ['initial_radial_grid_points', 'initial_axial_grid_points']
    for key in mesh_keys:
        if key not in config['mesh']:
            print(f"mesh 설정에 필수 키 '{key}'가 없습니다.")
            return False

    # simulation 설정 검사
    sim_keys = ['total_steps', 'dt', 'save_interval']
    for key in sim_keys:
        if key not in config['simulation']:
            print(f"simulation 설정에 필수 키 '{key}'가 없습니다.")
            return False

    # chemistry 설정 검사
    chem_keys = ['transport', 'reactions']
    for key in chem_keys:
        if key not in config['chemistry']:
            print(f"chemistry 설정에 필수 키 '{key}'가 없습니다.")
            return False

    # circuit 설정 검사
    circuit_keys = ['coil_radius', 'coil_height', 'coil_turns']
    for key in circuit_keys:
        if key not in config['circuit']:
            print(f"circuit 설정에 필수 키 '{key}'가 없습니다.")
            return False

    # constants 설정 검사
    const_keys = ['k', 'e', 'me', 'eps0']
    for key in const_keys:
        if key not in config['constants']:
            print(f"constants 설정에 필수 키 '{key}'가 없습니다.")
            return False

    return True

def main():
    """메인 실행 함수"""
    print("\n설정 파일 로드 중...")
    config = load_config()
    
    print("설정 유효성 검사 중...")
    validate_config(config)
    
    # 결과 저장 경로 설정
    results_dir = os.path.join('results', datetime.now().strftime('%Y%m%d_%H%M%S'))
    print(f"결과 저장 경로: {results_dir}")
    
    print("자기 일관성 해 계산 시작...")
    
    # 메시 초기화
    mesh = Mesh(config)
    
    # 모델 초기화
    plasma_model = PlasmaModel(config)
    circuit_model = CircuitModel(config)
    em_model = EMModel(config, mesh)
    
    # 자기 일관성 해결기 초기화
    solver = SelfConsistentSolver(plasma_model, circuit_model, em_model, mesh, config)
    
    # 시뮬레이션 실행
    n_steps = config['simulation']['total_steps']
    dt = config['simulation']['dt']
    solver.solve(config['process']['simulation_time_s'], dt, plot_interval=100, save_dir=results_dir)

if __name__ == "__main__":
    main()