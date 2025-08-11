import numpy as np
from numpy.polynomial.legendre import leggauss as gaussquad
from scipy.interpolate import _bspl as bspl
import matplotlib.pyplot as plt
import scipy.integrate
def create_ref_data(neval, deg, integrate=False):
    # reference unit domain
    reference_element = np.array([0, 1])
    if integrate is False:
        # point for plotting are equispaced on reference element
        x = np.linspace(reference_element[0], reference_element[1], neval)
        evaluation_points = x
        quadrature_weights = np.zeros((0,))
    else:
        # points (and weights) for integration are computed according to Gauss quadrature
        x, w = gaussquad(neval)
        evaluation_points = 0.5*(x + 1)
        quadrature_weights = w/2
    # knots for defining B-splines
    knt = np.concatenate((np.zeros((deg+1,),dtype=float),np.ones((deg+1,),dtype=float)),axis=0)
    # reference basis function values
    tmp = [bspl.evaluate_all_bspl(knt, deg, evaluation_points[i], deg, nu=0)
           for i in range(evaluation_points.shape[0])]
    reference_basis = np.vstack(tmp).T
    # reference basis function first derivatives
    tmp = [bspl.evaluate_all_bspl(knt, deg, evaluation_points[i], deg, nu=1)
           for i in range(evaluation_points.shape[0])]
    reference_basis_derivatives = np.vstack(tmp).T
    # store all data and return
    reference_data = {'reference_element': reference_element,
                      'evaluation_points': evaluation_points,
                      'quadrature_weights': quadrature_weights,
                      'deg': deg,
                      'reference_basis': reference_basis,
                      'reference_basis_derivatives': reference_basis_derivatives
    }
    return reference_data

def create_fe_space(deg, reg, mesh):
    def bezier_extraction(knt, deg):
        # breakpoints
        brk = np.unique(knt)
        # number of elements
        nel = brk.shape[0]-1
        # number of knots
        m = knt.shape[0]
        # assuming an open knotvector, knt[a] is the last repetition of the first knot
        a = deg
        # next knot
        b = a+1
        # Bezier element being processed
        nb = 0
        # first extraction matrix
        C = [np.eye(deg+1,deg+1, dtype=float)]
        # this is where knot-insertion coefficients are saved
        alphas = np.zeros((deg-1,),dtype=float)
        while b < m:
            # initialize extraction matrix for next element
            C.append(np.eye(deg+1,deg+1))
            # save index of current knot
            i = b
            # find last occurence of current knot
            while b < m-1 and knt[b+1] == knt[b]:
                b += 1
            # multiplicity of current knot
            mult = b-i+1
            # if multiplicity is < deg, smoothness is at least C0 and extraction may differ from an identity matrix
            if mult < deg:
                numer = knt[b] - knt[a]
                # smoothness of splines
                r = deg - mult
                # compute linear combination coefficients
                for j in range(deg-1,mult-1,-1):
                    alphas[j-mult] = numer / (knt[a+j+1]-knt[a])
                for j in range(r):
                    s = mult+j
                    for k in range(deg,s,-1):
                        alpha = alphas[k-s-1]
                        C[nb][:,k] = alpha*C[nb][:,k] + (1.0-alpha)*C[nb][:,k-1]
                    save = r-j
                    if b < m:
                        C[nb+1][save-1:j+save+1,save-1] = C[nb][deg-j-1:deg+1,deg]
            # increment element index
            nb += 1
            if b < m:
                a = b
                b += 1
            C = C[:nel]
        return C
    # number of mesh elements
    nel = mesh['m']
    # unique breakpoints
    if nel == 1:
        brk = mesh['elements'].T[0]
    else:
        brk = np.concatenate((mesh['elements'][0],
                              np.array([mesh['elements'][1][-1]])), axis=0)
    # multiplicity of each breakpoint
    mult = deg - reg
    # knot vector for B-spline definition
    knt = np.concatenate((np.ones((deg+1,), dtype=float) * brk[0],
                          np.ones((deg+1,), dtype=float) * brk[-1],
                          np.repeat(brk[1:-1],mult)), axis=0)
    knt = np.sort(knt)
    # coefficients of linear combination
    C = bezier_extraction(knt, deg)
    # dimension of finite element space
    dim = knt.shape[0]-deg-1
    # connectivity information (i.e., which bases are non-zero on which element)
    econn = np.zeros((nel,deg+1), dtype=int)
    for i in range(nel):
        if i == 0:
            econn[i] = np.arange( deg+1)
        else:
            econn[i] = econn[i-1] + mult
    # save and return
    space = {'n': dim,
             'supported_bases': econn,
             'extraction_coefficients': C
    }
    return space

def evaluate_solution(p,k,neval,u, x):
    mesh = create_mesh(brk = x)
    param_map = create_param_map(mesh = mesh)
    fe_space = create_fe_space(deg = p, reg = k, mesh = mesh)
    ref_data = create_ref_data(neval = neval, deg = p, integrate= False)

    reference_basis = ref_data['reference_basis']
    reference_basis_derivatives =  ref_data['reference_basis_derivatives']

    n = fe_space['n']
    extraction_coefficients = fe_space['extraction_coefficients']
    supported_bases = fe_space['supported_bases']
    elements = mesh['elements']
    m = mesh['m']

    
    x_eval = np.zeros([mesh['m'], neval])
    u_eval = np.zeros([mesh['m'], neval])
    du_dx_eval = np.zeros([mesh['m'], neval])
    du_dx = param_map['imap_derivatives']

    for element_index, element in enumerate(mesh['elements'].T):
        x_eval[element_index, :] = np.linspace(element[0], element[-1], neval)
        u_eval[element_index, :] = u[supported_bases[element_index]] @ (extraction_coefficients[element_index][:][:] @ reference_basis)
        du_dx_eval[element_index, :] = u[supported_bases[element_index]] @ (extraction_coefficients[element_index][:][:] @ reference_basis_derivatives) * du_dx[element_index]

    return x_eval.flatten(), u_eval.flatten(), du_dx_eval.flatten()

def create_mesh(brk):
    m = np.shape(brk)[0] - 1 #number of elements on our domian 
    elements = np.zeros((2, m)) #boundary values of each element
    elements[0] = brk[:-1]
    elements[1] = brk[1:]
    mesh = {'m':m, 'elements':elements}
    return mesh

def create_param_map(mesh):
    map = lambda psi, x0, x1: x0 + psi*(x1-x0)
    map_derivatives = mesh['elements'][1,:] - mesh['elements'][0,:]
    imap_derivatives = map_derivatives**-1
    param_map = {'map':map, 'map_derivatives':map_derivatives, 'imap_derivatives':imap_derivatives}
    return param_map    

def assemble_fe_problem(mesh, space ,ref_data, param_map, problem_B, problem_L, f, a_fourier, bc):
    evaluation_points = ref_data['evaluation_points']
    quadrature_weight = ref_data['quadrature_weights']

    extraction_coefficients = space['extraction_coefficients']

    reference_basis = ref_data['reference_basis']
    reference_basis_derivatives = ref_data['reference_basis_derivatives']

    n = space['n']
    A = np.zeros((n,n))
    b = np.zeros(n)
    neval = len(evaluation_points)
    for l in range(0,mesh['m']):
        supported_bases = space['supported_bases'][l]
        element = mesh['elements'][:,l]
        x_l = np.linspace(element[0], element[-1], neval)
        for i, basis_i in enumerate(supported_bases):
            N_i = np.dot(extraction_coefficients[l][i, :], reference_basis)
            dN_i = np.dot(extraction_coefficients[l][i, :], reference_basis_derivatives * param_map['imap_derivatives'][l])
            for j, basis_j in enumerate(supported_bases):
                N_j = np.dot(extraction_coefficients[l][j, :], reference_basis)
                dN_j = np.dot(extraction_coefficients[l][j, :], reference_basis_derivatives * param_map['imap_derivatives'][l])
                value = 0
                for r in range(0, neval):
                    value += problem_B(x_l[r], N_i[r], dN_i[r], N_j[r], dN_j[r]) * param_map['map_derivatives'][l] * quadrature_weight[r] 
                A[basis_i,basis_j] += value
            value = 0 
            for r in range(0,neval):
                value +=  problem_L(x_l[r], N_i[r], dN_i[r], f, a_fourier) * param_map['map_derivatives'][l] * quadrature_weight[r] 
            
            b[basis_i] += value 
        
    b+= -bc[0]*A[:,0] -bc[1]*A[:,-1] 
    return A[1:-1,1:-1], b[1:-1]

def problem_B1(x,Nj,dNj,Nk,dNk):
    return dNj * dNk

def problem_L1(x,Nj,dNj):
    return np.pi**2 * np.sin(np.pi*x) * Nj  

def problem_L(x,Nj,dNj):
    return  Nj  

def problem_B2(x,Nj,dNj,Nk,dNk):
    return (1-0.4*np.cos(np.pi * x))*dNj * dNk + Nj * Nk

def problem_L2(x,Nj,dNj):
    return  np.pi**2*np.sin(np.pi**2*x) * np.where(x < 1, 1, -1) * Nj





def projection(mesh, space, ref_data, param_map, u0 ):
    reference_basis = ref_data['reference_basis']
    extraction_coefficients = space['extraction_coefficients']
    reference_basis_derivatives = ref_data['reference_basis_derivatives']
    N = param_map['map'](ref_data['reference_basis'].T, mesh['elements'][0][0], mesh['elements'][0][1])
    alpha = []
    beta = 0
    A = np.zeros((space['n'], space['n']))
    
    
    b = np.zeros(space['n'])
    evaluation_points = ref_data['evaluation_points']
    neval = len(evaluation_points)
    for l in range(0,mesh['m']):
        supported_bases = space['supported_bases'][l]
        element = mesh['elements'][:,l]
        x_l = np.linspace(element[0], element[-1], neval)
        for i, basis_i in enumerate(supported_bases):
            N_i = np.dot(extraction_coefficients[l][i, :], reference_basis)
            dN_i = np.dot(extraction_coefficients[l][i, :], reference_basis_derivatives )
            for j, basis_j in enumerate(supported_bases):
                N_j = np.dot(extraction_coefficients[l][j, :], reference_basis)
                dN_j = np.dot(extraction_coefficients[l][j, :], reference_basis_derivatives * param_map['imap_derivatives'][l])
                A[basis_i,basis_j] += scipy.integrate.simpson(N_i*N_j)*param_map['map_derivatives'][l]
            b[basis_i] += scipy.integrate.simpson(N_i * u0(param_map['map'](ref_data['evaluation_points'], element[0], element[1]))) * param_map['map_derivatives'][l]
    
    return np.linalg.solve(A,b)
