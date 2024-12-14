
from ufl import sym, grad, inner, Measure, TestFunction, TrialFunction, SpatialCoordinate, as_vector, as_matrix
from basix.ufl import element

from dolfinx import fem, io
import dolfinx.fem.petsc
from dolfinx.mesh import create_rectangle, CellType

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np


thickness, inner_radius, outer_radius = 1, 2, 10
Nx, Ny = 5, 50
domain = create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, inner_radius]), np.array([thickness, outer_radius])],
    [Nx, Ny],
    cell_type=CellType.quadrilateral,
)


degree = 2
shape = (2,) 
V = fem.functionspace(domain, ("P", degree, shape))

u_sol = fem.Function(V, name="Displacement")


E = fem.Constant(domain, 210e3)
nu = fem.Constant(domain, 0.3)

x = SpatialCoordinate(domain)

def epsilon(v):
    e = sym(grad(v))
    e_theta = v[1] / x[1]


    return as_vector([e[1, 1],  # e_r
                      e[0, 0],  # e_z
                      e_theta,  # e_theta
                      e[0, 1]]) # gamma_rz


def sigma(v):

    C = E * as_matrix([[1-nu, nu, nu, 0], 
                       [nu, 1-nu, nu, 0], 
                       [nu, nu, 1-nu, 0], 
                       [0, 0, 0, (1-2*nu)/2]])

    return C * epsilon(v)

u = TrialFunction(V)
v = TestFunction(V)

rho = 2e-3
omega = 10
f = as_vector([0, rho * omega**2 * x[1]])

dx = Measure("dx", domain=domain)
a = inner(sigma(u), epsilon(v)) * dx
L = inner(f, v) * dx

def corner(x):
    return np.isclose(x[0], 0) & np.isclose(x[1], 0)

corner_dofs = fem.locate_dofs_geometrical(V, corner)

bcs = [
    fem.dirichletbc(PETSc.ScalarType(0), corner_dofs, V.sub(0)),    
]

problem = fem.petsc.LinearProblem(
    a, L, u=u_sol, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
problem.solve()

vtk = io.VTKFile(domain.comm, "linear_elasticity.pvd", "w")
vtk.write_function(u_sol)
vtk.close()

def as_tensor(X):
    return as_matrix([[X[1], X[3],  0  ],
                      [X[3], X[0],  0  ],
                      [ 0  ,  0  , X[2]]])


v_dg_el = element("DG", domain.basix_cell(), degree, shape=(3,3), dtype=PETSc.RealType)
W = fem.functionspace(domain, v_dg_el)
s_dg = fem.Function(W)
s_expr = fem.Expression(as_tensor(sigma(u_sol)), W.element.interpolation_points())
s_dg.interpolate(s_expr)

                     
vtk = io.VTKFile(domain.comm, "linear_elasticity_s.pvd", "w")
vtk.write_function(s_dg)
vtk.close()


s_expr = fem.Expression(as_tensor(epsilon(u_sol)), W.element.interpolation_points())
s_dg.interpolate(s_expr)

vtk = io.VTKFile(domain.comm, "linear_elasticity_e.pvd", "w")
vtk.write_function(s_dg)
vtk.close()

