import numpy as np
import math
from fipy import CellVariable, TransientTerm, DiffusionTerm, FaceVariable, ImplicitSourceTerm, CylindricalGrid2D

class CircuitModel:
    def __init__(self, config: dict):
        '''
        Circuit model

        Args:
            config: Configuration dictionary
        '''
        self.config = config
        self.Rp = 0

        r_max = self.config['coil']['coil_radius_max']
        r_min = self.config['coil']['coil_radius_min']
        coil_width = self.config['coil']['coil_width']
        N_turns = self.config['coil']['number_of_turns']

        self.r_coil = [r_min + i * coil_width + (2*i+1)*coil_width/2 for i in range(N_turns)]

        self.coil_length = [2*math.pi*r for r in self.r_coil].sum()
        self.Rc = config['coil']['coil_resistance_ohm'] * self.coil_length / coil_width**2 / math.pi

    def calculate_plasma_resistance(self, sigma_p: CellVariable, E_field: CellVariable) -> CellVariable:
        '''
        Calculate plasma resistance

        Args:
            sigma_p: Plasma conductivity
            E_field: Electric field

        Returns:
            Rp: Plasma resistance
        '''
        self.Rp = 1 / (sigma_p * E_field)
        return self.Rp
    
    def calculate_coil_current(self) -> float:
        '''
        Calculate coil current

        Returns:
            I: Coil current
        '''
        return math.sqrt(self.config['process']['input_power_W']/(self.Rp + self.Rc))

class EMModel:
    def __init__(self, config: dict, mesh: CylindricalGrid2D):
        '''
        EM model

        Args:
            config: Configuration dictionary
            mesh: Mesh
        '''
        self.config = config
        self.mesh = mesh

    def set_coil_current(self, I: float) -> FaceVariable:
        '''
        Set coil current

        Args:
            I: Coil current

        Returns:
            J_coil: Coil current
        '''
        I_peak = I * math.sqrt(2)

        r_max = self.config['coil']['coil_radius_max']
        r_min = self.config['coil']['coil_radius_min']
        coil_width = self.config['coil']['coil_width']
        N_turns = self.config['coil']['number_of_turns']

        coil_spacing = (r_max - r_min - coil_width*N_turns) / (N_turns - 1)
        coil_position = []

        for i in range(N_turns):
            coil_position.append(r_min + i * coil_spacing + (2*i+1)*coil_width/2)
        
        self.coil_current = I_peak * N_turns / (r_max - r_min)

        J_coil = FaceVariable(mesh=self.mesh, value=0.0+0.0j, name='J_coil')

        for coil_p in coil_position:
            J_coil.constrain(self.coil_current, where=self.mesh.getCellCenters()[0] == coil_p)
        return J_coil
    
    def solve_helmholtz_equation(self, sigma_p: CellVariable, J_coil: FaceVariable) -> CellVariable:
        '''
        Solve helmholtz equation

        Args:
            sigma_p: Plasma conductivity
            J_coil: Coil current

        Returns:
            A_theta: Magnetic vector potential
        '''

        omega = 2 * math.pi * self.config['process']['radio_frequency_Hz']
        mu0 = self.config['constants']['mu0']
        k2 = CellVariable(mesh=self.mesh, value = 1j*omega**2 * mu0 * sigma_p.value)

        A = CellVariable(mesh=self.mesh, value=0.0)
        A.constrain(0.0, self.mesh.facesRight)
        A.constrain(0.0, self.mesh.faceBottom)

        eq = (DiffusionTerm(coeff=self.mesh.r) + self.mesh.r * ImplicitSourceTerm(coeff=k2) == self.mesh.r * mu0 * J_coil)
        eq.solve(var=A)

        A_theta = A.value.reshape(self.mesh.nz, self.mesh.nr)

        return A_theta
    
class PlasmaModel:
    def __init__(self, config: dict, mesh: CylindricalGrid2D):
        '''
        Plasma model

        Args:
            config: Configuration dictionary
            mesh: Mesh
        '''
        self.config = config
        self.mesh = mesh


    def solve_continuity_equation(
            self, 
            ng: CellVariable, ne: CellVariable, ni: CellVariable, nms: CellVariable, 
            E_field: CellVariable, B_field: CellVariable,
            ) -> CellVariable:
        '''
        Solve continuity equation
        '''

        dne, dni, dng, dms = self.cal_reactions(ne, ni, ng, nms)
        j_e, j_i = self.cal_transport(ne, ni, E_field, B_field)

        dne += j_e
        dni += j_i

        ne = ne + dne
        ni = ni + dni
        ng = ng + dng
        nms = nms + dms

        return ne, ni, ng, nms

    def solve_continuity_equation(
            self, ng: CellVariable, ne: CellVariable, ni: CellVariable, nms: CellVariable, 
            E_field: CellVariable, B_field: CellVariable,
            ) -> CellVariable:
        '''
        Solve continuity equation

        Args:
            ng: Neutral gas density
            ne: Electron density
            ni: Ion density
            nms: Neutral gas density
            E_field: Electric field
            B_field: Magnetic field
        '''
        pass

    def cal_transport(
            self,
            ne : CellVariable, ni: CellVariable, 
            E_field: CellVariable, B_field: CellVariable
            ) -> CellVariable:
        '''
        Calculate transport
        '''

        D_e = self.config['chemistry']['transport']['D_e']
        D_i = self.config['chemistry']['transport']['D_i']
        mu_e = self.config['chemistry']['transport']['mu_e']
        mu_i = self.config['chemistry']['transport']['mu_i']

        j_e = -D_e * ne.cellGrad.value[0] - mu_e * E_field.value * ne
        j_i = -D_i * ni.cellGrad.value[0] - mu_i * E_field.value * ni

        j_e = j_e.reshape(self.mesh.nz, self.mesh.nr)
        j_i = j_i.reshape(self.mesh.nz, self.mesh.nr)

        return j_e, j_i

    
    def cal_heating(
            self,
            ne : CellVariable, ni: CellVariable, 
            E_field: CellVariable, B_field: CellVariable
            ) -> CellVariable:
        '''
        Calculate heating
        '''

        E_e = self.config['chemistry']['heating']['E_e']
        E_i = self.config['chemistry']['heating']['E_i']
        
    
    def cal_reactions(
            self,
            Te: CellVariable, Tg: CellVariable, 
            ng: CellVariable, ne: CellVariable, ni: CellVariable, nms: CellVariable
            ) -> CellVariable:
        '''
        Calculate reactions
        '''

        kiz = self.config['chemistry']['reactions']['ionization']['k0'] * math.exp(-self.config['chemistry']['reactions']['ionization']['E_ion_J'] / (self.config['constants']['k'] * Te))
        kstep = self.config['chemistry']['reactions']['step_ionization']['k0'] * math.exp(-self.config['chemistry']['reactions']['step_ionization']['E_step_J'] / (self.config['constants']['k'] * Te))
        kex = self.config['chemistry']['reactions']['excitation']['k0'] * math.exp(-self.config['chemistry']['reactions']['excitation']['E_exc_J'] / (self.config['constants']['k'] * Te))
        kre = self.config['chemistry']['reactions']['recombination']['radiative']['k0'] * math.exp(-self.config['chemistry']['reactions']['recombination']['radiative']['E_re_J'] / (self.config['constants']['k'] * Te))
        kri = self.config['chemistry']['reactions']['recombination']['three_body']['k0'] * math.exp(-self.config['chemistry']['reactions']['recombination']['three_body']['E_re_J'] / (self.config['constants']['k'] * Te))

        dne = kiz * ng * ne - kex * ne * ng - kstep * ne * nms + kre * ne**2 + kri * ne * ni
        dni = kiz * ng * ne + kstep * ne * nms - kre * ni * ne - kri * ni * ne**2
        dng = -kiz * ng * ne - kex * ne * ng + kre * ni * ne + kri * ne**2 * ni
        dms = kex * ne * ng - kstep * ne * nms

        return dne, dni, dng, dms
    
    def solve(self,
              ) -> CellVariable:
        '''
        Solve continuity equation
        '''

        self.cal_heating()
        ne, ni, ng, nms = self.solve_continuity_equation(ng, ne, ni, nms, E_fild, B_field)
        return ne, ni, ng, nms
    
    