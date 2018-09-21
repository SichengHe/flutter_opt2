# This is a template that should be used for setting up
# RANS analysis scripts

# ======================================================================
#         Import modules
# ======================================================================
import numpy as np
from mpi4py import MPI
from baseclasses import *
import argparse
import copy



complex_flag = False

# alpha and mach
alpha = 1.0000
mach = 0.73
if 1: 
    if complex_flag:
        # mach = complex(0.73, 1e-40)
        alpha = complex(alpha, 1e-40)
    else:
        pass

# shape x
ncoef = 8
array0 = []

for i in xrange(ncoef):
    array0.append(0.0)

array0 = np.array(array0)

if 0:
    if complex_flag:
        array0[0] = complex(array0[0], 1e-40)
    else:
        pass

# import complex modules / real modules

if complex_flag:

    from adflow import ADFLOW_C
    from pywarpustruct import USMesh_C

else:

    from adflow import ADFLOW
    from pywarpustruct import USMesh 



# complex step sin cos functions:
def csin(x):
    "complex sin"
    x_real = np.real(x)
    x_imag = np.imag(x)

    return np.sin(x_real)*np.cosh(x_imag) + 1j*np.cos(x_real)*np.sinh(x_imag)


def ccos(x):
    "complex cos"
    x_real = np.real(x)
    x_imag = np.imag(x)

    return np.cos(x_real)*np.cosh(x_imag) - 1j*np.sin(x_real)*np.sinh(x_imag)






# complex step 
def setVolumeGrid(structU, ntimeintervalsspectral, b, xRot, aeroSolver, cfdPoints, complex_flag, perturb_flag):

    # this is a different interface compared with other TACs based method!

    # get the volume grid for all time instances from structure displacement
    # NOTICE: dimensional displacement should be used

    N_pts = cfdPoints.shape[0]

    for i in xrange(ntimeintervalsspectral):

        if not complex_flag:
            new_coords = np.zeros((N_pts, 3))
        else:
            new_coords = []
            for j in xrange(N_pts):
                new_coords.append([0j, 0j, 0j])

            new_coords = np.array(new_coords)

        plg_loc = structU[2*i]*b # positive downwards
        ptch_loc = structU[2*i + 1]

        if complex_flag:

            cc = ccos(ptch_loc)
            ss = csin(ptch_loc)

        else:

            cc = np.cos(ptch_loc)
            ss = np.sin(ptch_loc)

        for j in xrange(N_pts):

            new_coords[j, 0] = (  cc*(cfdPoints[j, 0] - xRot) + ss*cfdPoints[j, 1] + xRot)
            new_coords[j, 1] = (- ss*(cfdPoints[j, 0] - xRot) + cc*cfdPoints[j, 1] - plg_loc)
            new_coords[j, 2] = cfdPoints[j, 2]


        aeroSolver.mesh.setSurfaceCoordinates(new_coords)

        aeroSolver.mesh.warpMesh()

        m = aeroSolver.mesh.getSolverGrid()

        aeroSolver.adflow.warping.setgridforoneinstance(m,sps=i+1)




name = 'adjoint_test'
outputDirectory = './'
gridFile = './naca64A010_euler-L2.cgns'
FFDFile = './ffd.xyz'

areaRef = 1.0
chordRef = 1.0


alpha_mag = 1.0/180*np.pi
omega = 80.0

b = 0.5
xRot = 0.248
yRot = 0.0
xRef = 0.248
yRef = 0.0

ntimeintervalsspectral = 3



# generate displacement
structU = np.zeros(2*ntimeintervalsspectral)
for i in xrange(ntimeintervalsspectral):
    phase = np.float(i)/3.0*(2.0*np.pi)
    structU[2*i + 1] = -np.sin(phase)*alpha_mag


aeroOptions = {
    # Common Parameters
    'gridFile':gridFile,
    'outputDirectory':outputDirectory,

    # Physics Parameters
    'equationType':'euler',
    'smoother':'dadi',

    'equationMode':'time spectral',

    'CFL':0.7,
    'CFLCoarse':0.3,
    'MGCycle':'sg',
    'MGStartLevel':-1,
    'nCyclesCoarse':3000,
    'nCycles':20000,
    'monitorvariables':['resrho','cl','cd','cmz','yplus','resturb'],
    'surfaceVariables':['vx','vy','cp','mach','cf','cfx','cfy','rho','p','temp','mach','ptloss'],
    'useNKSolver':True,
    # 'useanksolver' : True,
    'nsubiterturb' : 10,
    'liftIndex':2,
    # Convergence Parameters
    'L2Convergence':1e-15,
    'L2ConvergenceCoarse':1e-5,
    # Adjoint Parameters
    'adjointSolver':'gmres', #gmres,tfqmr,rechardson,bcgs,ibcgs
    'adjointL2Convergence':1e-12,
    'ADPC':True,
    #'ADPC':False, #hxl
    'adjointMaxIter': 8000,
    'adjointSubspaceSize':400,
    'ILUFill':3,
    #'ILUFill':2, #hxl
    'ASMOverlap':3,
    'outerPreconIts':3,
    #'innerPreconIts':2, #hxl
    'NKSubSpaceSize':400,
    'NKASMOverlap':4,
    'NKPCILUFill':4,
    'NKJacobianLag':5,
    'nkswitchtol':1e-2, #2e-4,
    'nkouterpreconits':3,
    'NKInnerPreConIts':3,
    'writeSurfaceSolution':False,
    'writeVolumeSolution':True,
    'frozenTurbulence':False,
    'restartADjoint':False, 
    # "usematrixfreedrdw": False,

    'timeIntervals': ntimeintervalsspectral,
    'qmode':True,
    'alphaFollowing': False,
    'TSStability': False,
    'usetsinterpolatedgridvelocity': True,

    'useblockettes': False, 
    }


usoptions = {
  'gridFile':gridFile,
  'fileType':'CGNS',
  # 'specifiedSurfaces':None,
  # 'symmetrySurfaces':None,
  # 'symmetryPlanes':[],
  # 'aExp': 3.0,
  # 'bExp': 5.0,
  # 'LdefFact':1.0,
  # 'alpha':0.25,
  # 'errTol':0.0001,
  # 'evalMode':'fast',
  # 'useRotations':True,
  # 'zeroCornerRotations':True,
  # 'cornerAngle':30.0,
  # 'bucketSize':8,
}

execfile('./setup_geometry.py')


DVList={'shape':array0}
DVGeo.setDesignVars(DVList)


if complex_flag:
    CFDSolver = ADFLOW_C(options=aeroOptions)
else:
    CFDSolver = ADFLOW(options=aeroOptions)

CFDSolver.setDVGeo(DVGeo)

ap = AeroProblem(name=name, alpha=alpha, mach=mach, altitude=10000.0, areaRef=1.0, chordRef=1.0, evalFuncs=['cl','cd', 'cmz'],
    xRef=xRef,xRot=xRot,\
    degreePol=0,coefPol=[0.0],\
    degreeFourier=1,omegaFourier=omega,cosCoefFourier=[0.0,0.0],sinCoefFourier=[0.0])


ap.addDV('mach',scale=1.0)
ap.addDV('alpha',scale=1.0)


if complex_flag:
    mesh=USMesh_C(options=usoptions)
else:
    mesh=USMesh(options=usoptions)

CFDSolver.setMesh(mesh)




funcs = {}
# CFDSolver(ap)
CFDSolver.setAeroProblem(ap)
cfdPoints = CFDSolver.getInitialSurfaceCoordinates(CFDSolver.allWallsGroup)

# print "cfdPoints", cfdPoints
# exit()
















if 0: 

    setVolumeGrid(structU, ntimeintervalsspectral, b, xRot, CFDSolver, cfdPoints, complex_flag, False)
    CFDSolver(ap)
    setVolumeGrid(structU, ntimeintervalsspectral, b, xRot, CFDSolver, cfdPoints, complex_flag, True)

    index_r = 3881 - 1 # rho
    residual1 = CFDSolver.getResidual(ap)

    print("res rho", residual1[index_r])
    print("res ivx", residual1[index_r+1])
    print("res ivy", residual1[index_r+2])
    print("res ivz", residual1[index_r+3])




























if 1:

    setVolumeGrid(structU, ntimeintervalsspectral, b, xRot, CFDSolver, cfdPoints, complex_flag, False)
    CFDSolver(ap)



# surgery room 1
# local section
# inviscidcentralflux:
if 0:
    # forward pass

    CFDSolver.hsc_intializedSeed()

    fd_seed = np.random.rand(1, 3, 24, 16, 1)

    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            for i in range(2, 25+1):
                for j in range(2, 17+1):
                    for k in range(2, 2+1):

                        fd_seed_loc = fd_seed[mm-1, sps-1, i-2, j-2, k-2]

                        CFDSolver.hsc_setsfaceid(mm, sps, i, j, k, fd_seed_loc)

    for mm in range(1, 1+1):
        for sps in range(1, 3+1):

            CFDSolver.hsc_forwardad(mm, sps)


    pfpx_fd = np.zeros((1, 3, 24, 16, 1, 5))

    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            for i in range(2, 25+1):
                for j in range(2, 17+1):
                    for k in range(2, 2+1):
                        for ii in range(1, 5+1):

                            pfpx_fd[mm-1, sps-1, i-2, j-2, k-2, ii-1] = CFDSolver.hsc_getsfaceid(mm, sps, i, j, k, ii)

    CFDSolver.hsc_deallocateSeed()
    



    # reverse pass
    # it takes the seed from the forward pass

    CFDSolver.hsc_intializedSeed()

    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            for i in range(2, 25+1):
                for j in range(2, 17+1):
                    for k in range(2, 2+1):
                        for ii in range(1, 5+1):

                            rd_seed_loc = pfpx_fd[mm-1, sps-1, i-2, j-2, k-2, ii-1]

                            CFDSolver.hsc_setresd(mm, sps, i, j, k, ii, rd_seed_loc)
                            
    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            CFDSolver.hsc_reversead(mm, sps)

    pfpx_rd = np.zeros((1, 3, 24, 16, 1))


    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            for i in range(2, 25+1):
                for j in range(2, 17+1):
                    for k in range(2, 2+1):

                        pfpx_rd[mm-1, sps-1, i-2, j-2, k-2] = CFDSolver.hsc_getresd(mm, sps, i, j, k)


    CFDSolver.hsc_deallocateSeed()

    # comparison


    if MPI.COMM_WORLD.rank == 0:


        fd_seed = fd_seed.reshape((1*3*24*16*1))
        pfpx_rd = pfpx_rd.reshape((1*3*24*16*1))

        pfpx_fd = pfpx_fd.reshape((1*3*24*16*1*5))

        startend_prod = np.dot(fd_seed, pfpx_rd)
        midmid_prod = np.dot(pfpx_fd, pfpx_fd)


        print("startend_prod", startend_prod, "midmid_prod", midmid_prod)






# initRes_block
if 0:

    # forward pass

    CFDSolver.hsc_intializedSeed()

    fd_seed = np.random.rand(1, 3, 24, 16, 1)


    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            for i in range(2, 25+1):
                for j in range(2, 17+1):
                    for k in range(2, 2+1):

                        fd_seed_loc = fd_seed[mm-1, sps-1, i-2, j-2, k-2]

                        CFDSolver.hsc_setvolumed(mm, sps, i, j, k, fd_seed_loc)

    for mm in range(1, 1+1):
        for sps in range(1, 3+1):

            CFDSolver.hsc_initres_fd(mm, sps)

    pfpx_fd = np.zeros((1, 3, 24, 16, 1, 5))

    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            for i in range(2, 25+1):
                for j in range(2, 17+1):
                    for k in range(2, 2+1):
                        for ii in range(1, 5+1):

                            pfpx_fd[mm-1, sps-1, i-2, j-2, k-2, ii-1] = CFDSolver.hsc_getdwd(mm, sps, i, j, k, ii)

    CFDSolver.hsc_deallocateSeed()


    # reverse pass
    # it takes the seed from the forward pass

    CFDSolver.hsc_intializedSeed()

    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            for i in range(2, 25+1):
                for j in range(2, 17+1):
                    for k in range(2, 2+1):
                        for ii in range(1, 5+1):

                            rd_seed_loc = pfpx_fd[mm-1, sps-1, i-2, j-2, k-2, ii-1]

                            CFDSolver.hsc_setdwd(mm, sps, i, j, k, ii, rd_seed_loc)
                            
    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            CFDSolver.hsc_initres_rd(mm, sps)

    pfpx_rd = np.zeros((1, 3, 24, 16, 1))


    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            for i in range(2, 25+1):
                for j in range(2, 17+1):
                    for k in range(2, 2+1):

                        pfpx_rd[mm-1, sps-1, i-2, j-2, k-2] = CFDSolver.hsc_getvolumed(mm, sps, i, j, k)


    CFDSolver.hsc_deallocateSeed()


    if MPI.COMM_WORLD.rank == 0:


        fd_seed = fd_seed.reshape((1*3*24*16*1))
        pfpx_rd = pfpx_rd.reshape((1*3*24*16*1))

        pfpx_fd = pfpx_fd.reshape((1*3*24*16*1*5))

        startend_prod = np.dot(fd_seed, pfpx_rd)
        midmid_prod = np.dot(pfpx_fd, pfpx_fd)


        print("startend_prod", startend_prod, "midmid_prod", midmid_prod)



if 0:

    
    # forward pass

    CFDSolver.hsc_intializedSeed()

    fd_seed = np.random.rand(1, 3, 24, 16, 1)


    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            for i in range(2, 25+1):
                for j in range(2, 17+1):
                    for k in range(2, 2+1):

                        fd_seed_loc = fd_seed[mm-1, sps-1, i-2, j-2, k-2]

                        CFDSolver.hsc_setsfaceid(mm, sps, i, j, k, fd_seed_loc)

    for mm in range(1, 1+1):
        for sps in range(1, 3+1):

            CFDSolver.hsc_timestep_block_fd(False, mm, sps)

    pfpx_fd = np.zeros((1, 3, 24, 16, 1))

    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            for i in range(2, 25+1):
                for j in range(2, 17+1):
                    for k in range(2, 2+1):

                        pfpx_fd[mm-1, sps-1, i-2, j-2, k-2] = CFDSolver.hsc_getradid(mm, sps, i, j, k)

    CFDSolver.hsc_deallocateSeed()

    # reverse pass
    # it takes the seed from the forward pass

    CFDSolver.hsc_intializedSeed()

    rd_seed = np.random.rand(1, 3, 24, 16, 1)

    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            for i in range(2, 25+1):
                for j in range(2, 17+1):
                    for k in range(2, 2+1):

                        # rd_seed_loc = pfpx_fd[mm-1, sps-1, i-2, j-2, k-2]

                        rd_seed_loc = rd_seed[mm-1, sps-1, i-2, j-2, k-2]
                        CFDSolver.hsc_setradid(mm, sps, i, j, k, rd_seed_loc)
                            
    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            CFDSolver.hsc_timestep_block_rd(False, mm, sps)

    pfpx_rd = np.zeros((1, 3, 24, 16, 1))


    for mm in range(1, 1+1):
        for sps in range(1, 3+1):
            for i in range(2, 25+1):
                for j in range(2, 17+1):
                    for k in range(2, 2+1):

                        pfpx_rd[mm-1, sps-1, i-2, j-2, k-2] = CFDSolver.hsc_getresd(mm, sps, i, j, k)


    CFDSolver.hsc_deallocateSeed()

    if MPI.COMM_WORLD.rank == 0:


        fd_seed = fd_seed.flatten()
        pfpx_rd = pfpx_rd.flatten()

        rd_seed = rd_seed.flatten()
        pfpx_fd = pfpx_fd.flatten()

        startend_prod = np.dot(fd_seed, pfpx_rd)
        midmid_prod = np.dot(rd_seed, pfpx_fd)


        print("startend_prod", startend_prod, "midmid_prod", midmid_prod)

    












    








# if 1:

#     print "##############################################################################################"
#     setVolumeGrid(structU, ntimeintervalsspectral, b, xRot, CFDSolver, cfdPoints, complex_flag, False)
#     CFDSolver.gridVelocitiesFineLevel_TS_block(1, 1)
#     # CFDSolver.gridVelocitiesFineLevel_TS_block_d(1, 1)
#     CFDSolver.gridVelocitiesFineLevel_TS_block_b(1, 1)

#     # CFDSolver.gridvelocitiesfinelevel_ts_block_d()
#     print "##############################################################################################"












# exit()


# surgery room 2
# major operation
if 1:
    if 1:

        
        CFDSolver.evalFunctions(ap, funcs)
        funcsSens = {}

        print("calculating sensitivity")
        CFDSolver.evalFunctionsSens(ap, funcsSens)



    if 1:
        if MPI.COMM_WORLD.rank == 0:

            if complex_flag:

                fobj=open('obj_new.txt','w')
                fobj.write('fc_cl: %.15f, fc_cd:%.15f, fc_cmz:%.15f \n'%(np.imag(funcs[name+'_cl']),np.imag(funcs['fc_cd']),np.imag(funcs['fc_cmz'])))

            if MPI.COMM_WORLD.rank == 0:
                fobj=open('obj_new.txt','w')
                fobj.write('fc_cl %.15f fc_cd %.15f fc_cmz %.15f \n'%(funcs['fc_cl'],funcs['fc_cd'],funcs['fc_cmz']))
                fobj.write('dfc_cl/dmach_fc %.15f \n'%(funcsSens['fc_cl']['mach_fc']))
                fobj.write('dfc_cl/dalpha_fc %.15f \n'%(funcsSens['fc_cl']['alpha_fc']))

                for i in xrange(ncoef):
                    fobj.write('dfc_cl/dx %.15f \n'%(funcsSens['fc_cl']['shape'][0][i]))




