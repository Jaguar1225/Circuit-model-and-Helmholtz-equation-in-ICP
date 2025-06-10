from fipy import CylindricalGrid2D, CellVariable, DiffusionTerm, TransientTerm

def create_mesh(config):
    R = config['chamber']['chamber_radius_m']
    Z = config['chamber']['chamber_height_m']
    Nr = config['mesh']['initial_radial_grid_points']
    Nz = config['mesh']['initial_axial_grid_points']

    dr = R / (Nr-1)
    dz = Z / (Nz-1)

    mesh = CylindricalGrid2D(dr=dr, dz=dz, nr=Nr, nz=Nz)
    phi = CellVariable(mesh=mesh, value=0.0)

    ng = CellVariable(mesh=mesh, value=0.0)
    ng.constrain(0.0, where=mesh.getCellCenters()[0] < R)

    ne = CellVariable(mesh=mesh, value=0.0)
    ni = CellVariable(mesh=mesh, value=0.0)
    nms = CellVariable(mesh=mesh, value=0.0)

    Te = CellVariable(mesh=mesh, value=0.0)
    sigma_p = CellVariable(mesh=mesh, value=0.0)


    return mesh, phi, ng, ne, ni, nms, Te, sigma_p

class Mesh:
    def __init__(self, config):
        self.mesh, self.phi, self.ng, self.ne, self.ni, self.nms, self.Te, self.sigma_p = create_mesh(config)

    def get_mesh(self):
        return self.mesh
    
    def get_phi(self):
        return self.phi
    
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