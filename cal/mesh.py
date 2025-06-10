import numpy as np

def create_mesh(config):
    """
    격자 생성 함수
    
    Returns:
        tuple: (mesh_r, mesh_z, ne, ni, ng, nms, Te, sigma_p, E_field, Rp)
        - mesh_r, mesh_z: 2D 격자 좌표
        - ne, ni, ng, nms: 복소수 밀도 배열 (shape: (nz, nr))
        - Te: 전자 온도 배열
        - sigma_p: 플라즈마 전도도 배열
        - E_field: 전기장 배열
        - Rp: 플라즈마 저항
    """
    R = config['chamber']['chamber_radius_m']
    Z = config['chamber']['chamber_height_m']
    Nr = config['mesh']['initial_radial_grid_points']
    Nz = config['mesh']['initial_axial_grid_points']

    # 격자 생성
    r = np.linspace(0, R, Nr)
    z = np.linspace(0, Z, Nz)
    mesh_r, mesh_z = np.meshgrid(r, z)

    # 초기값 설정 (복소수 지원)
    ne = np.full((Nz, Nr), config['initial']['electron_density_m3'] + 0j, dtype=complex)  # 전자 밀도
    ni = np.full((Nz, Nr), config['initial']['ion_density_m3'] + 0j, dtype=complex)  # 이온 밀도
    ng = np.full((Nz, Nr), config['initial']['neutral_gas_density_m3'] + 0j, dtype=complex) # 중성 가스 밀도
    nms = np.full((Nz, Nr), config['initial']['metastable_density_m3'] + 0j, dtype=complex) # 준안정 상태 밀도
    
    Te = np.full((Nz, Nr), config['initial']['electron_temperature_eV'])                       # 전자 온도
    sigma_p = np.full((Nz, Nr), config['initial']['plasma_conductivity_Sm'] + 0j, dtype=complex)  # 플라즈마 전도도
    E_field = np.full((Nz, Nr), config['initial']['electric_field_Vm'] + 0j, dtype=complex)  # 전기장
    Rp = config['initial']['plasma_resistance_ohm']                                          # 플라즈마 저항

    # 경계 조건 설정 (예: r=0에서 대칭)
    ne[:, 0] = ne[:, 1]  # r=0에서 대칭
    ni[:, 0] = ni[:, 1]
    ng[:, 0] = ng[:, 1]
    nms[:, 0] = nms[:, 1]
    Te[:, 0] = Te[:, 1]
    sigma_p[:, 0] = sigma_p[:, 1]
    E_field[:, 0] = E_field[:, 1]

    return mesh_r, mesh_z, ne, ni, ng, nms, Te, sigma_p, E_field, Rp

class Mesh:
    def __init__(self, config: dict):
        """
        격자 생성 및 초기화
        """
        self.config = config
        
        # 격자 생성 (더 촘촘하게)
        nr = config['mesh']['initial_radial_grid_points'] * 2  # 2배 더 촘촘하게
        nz = config['mesh']['initial_axial_grid_points'] * 2
        self.mesh_r, self.mesh_z = np.meshgrid(
            np.linspace(0, config['chamber']['chamber_radius_m'], nr),
            np.linspace(0, config['chamber']['chamber_height_m'], nz)
        )
        
        # 격자 간격
        self.dr = config['chamber']['chamber_radius_m'] / (nr - 1)
        self.dz = config['chamber']['chamber_height_m'] / (nz - 1)
        
        # 경계 조건 마스크 (더 엄격하게)
        self.boundary_mask = np.ones_like(self.mesh_r, dtype=bool)
        # 벽 경계 (r = R, z = H)에서 더 엄격하게 적용
        r_wall = config['chamber']['chamber_radius_m']
        z_wall = config['chamber']['chamber_height_m']
        dr_wall = 0.05 * self.dr  # 벽 근처 5% 구간
        dz_wall = 0.05 * self.dz
        self.boundary_mask[
            (self.mesh_r > (r_wall - dr_wall)) | (self.mesh_z > (z_wall - dz_wall))
        ] = False
        
        # 대칭 경계 (r = 0)에서 더 엄격하게 적용
        dr_sym = 0.1 * self.dr  # 대칭 경계 근처 10% 구간
        self.boundary_mask[self.mesh_r < dr_sym] = False

        # (config/initial.json)에서 초기값을 읽어와서 ne, ni, ng, nms, Te, sigma_p, E_field, Rp를 초기화
        initial = config['initial']
        self.ne = (initial['electron_density_m3'] * np.ones_like(self.mesh_r, dtype=complex))
        self.ni = (initial['ion_density_m3'] * np.ones_like(self.mesh_r, dtype=complex))
        self.ng = (initial['neutral_gas_density_m3'] * np.ones_like(self.mesh_r, dtype=float))
        self.nms = (initial['metastable_density_m3'] * np.ones_like(self.mesh_r, dtype=complex))
        self.Te = (initial['electron_temperature_eV'] * np.ones_like(self.mesh_r, dtype=float))
        self.sigma_p = (initial['plasma_conductivity_Sm'] * np.ones_like(self.mesh_r, dtype=complex))
        self.E_field = (initial['electric_field_Vm'] * np.ones_like(self.mesh_r, dtype=complex))
        self.Rp = (initial['plasma_resistance_ohm'])

        # (경계 조건 적용)
        self.apply_boundary_conditions()

    def get_mesh(self):
        return self.mesh_r, self.mesh_z
    
    def get_phi(self):
        return np.zeros_like(self.mesh_r)  # 필요시 구현
    
    def get_ng(self):
        return self.ng
    
    def get_ne(self):
        return self.ne
    
    def get_ni(self):
        return self.ni
    
    def get_nms(self):
        return self.nms
    
    def get_Te(self):
        return self.Te
    
    def get_sigma_p(self):
        return self.sigma_p
    
    def get_E_field(self):
        return self.E_field
    
    def get_Rp(self):
        return self.Rp

    def apply_boundary_conditions(self) -> None:
        """
        경계 조건 적용 (더 엄격하게)
        """
        # 벽 경계 (r = R, z = H)에서 더 엄격하게 적용
        r_wall = self.config['chamber']['chamber_radius_m']
        z_wall = self.config['chamber']['chamber_height_m']
        dr_wall = 0.05 * self.dr  # 벽 근처 5% 구간
        dz_wall = 0.05 * self.dz
        mask_wall = (self.mesh_r > (r_wall - dr_wall)) | (self.mesh_z > (z_wall - dz_wall))
        if hasattr(self, 'ne'):
            self.ne[mask_wall] = 0.0
        if hasattr(self, 'ni'):
            self.ni[mask_wall] = 0.0
        if hasattr(self, 'ng'):
            self.ng[mask_wall] = 0.0
        if hasattr(self, 'nms'):
            self.nms[mask_wall] = 0.0
        if hasattr(self, 'Te'):
            self.Te[mask_wall] = 0.0
        if hasattr(self, 'sigma_p'):
            self.sigma_p[mask_wall] = 0.0
        if hasattr(self, 'E_field'):
            self.E_field[mask_wall] = 0.0

        # 대칭 경계 (r = 0)에서 더 엄격하게 적용
        dr_sym = 0.1 * self.dr  # 대칭 경계 근처 10% 구간
        mask_sym = (self.mesh_r < dr_sym)
        if hasattr(self, 'ne'):
            self.ne[mask_sym] = 0.0
        if hasattr(self, 'ni'):
            self.ni[mask_sym] = 0.0
        if hasattr(self, 'ng'):
            self.ng[mask_sym] = 0.0
        if hasattr(self, 'nms'):
            self.nms[mask_sym] = 0.0
        if hasattr(self, 'Te'):
            self.Te[mask_sym] = 0.0
        if hasattr(self, 'sigma_p'):
            self.sigma_p[mask_sym] = 0.0
        if hasattr(self, 'E_field'):
            self.E_field[mask_sym] = 0.0
