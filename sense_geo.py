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

parser = argparse.ArgumentParser()
parser.add_argument("--flag_complex", help="bool var for complex to real implementation. True or False",
                   type=str, default=True)

args = parser.parse_args()

complex_flag = args.flag_complex
if complex_flag == 'False':
    complex_flag = False
else:
    complex_flag = True


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




name = 'adjoint_test'
outputDirectory = './OUTPUT'
gridFile = './INPUT/naca64A010_euler-L2.cgns'
FFDFile = './INPUT/ffd.xyz'

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
    'usetsinterpolatedgridvelocity': False,

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

execfile('./utils/setup_geometry.py')


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
    degreeFourier=1,omegaFourier=omega,cosCoefFourier=[0.0,0.0],sinCoefFourier=[alpha_mag])


ap.addDV('mach',scale=1.0)
ap.addDV('alpha',scale=1.0)


if complex_flag:
    mesh=USMesh_C(options=usoptions)
else:
    mesh=USMesh(options=usoptions)

CFDSolver.setMesh(mesh)


funcs = {}
CFDSolver.setAeroProblem(ap)
CFDSolver(ap)


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

                fobj=open(outputDirectory + '/'+ 'obj_geo.txt','w')
                fobj.write('fc_cl: %.15f, fc_cd:%.15f, fc_cmz:%.15f \n'%(np.imag(funcs[name+'_cl']),np.imag(funcs[name+'_cd']),np.imag(funcs[name+'_cmz'])))

            if MPI.COMM_WORLD.rank == 0:
                fobj=open(outputDirectory + '/'+ 'obj_geo.txt','w')
                fobj.write('cl %.15f cd %.15f cmz %.15f \n'%(funcs[name+'_cl'],funcs[name+'_cd'],funcs[name+'_cmz']))
                fobj.write('dcl/dmach %.15f \n'%(funcsSens[name+'_cl']['mach_'+name]))
                fobj.write('dcl/dalpha %.15f \n'%(funcsSens[name+'_cl']['alpha_'+name]))

                for i in xrange(ncoef):
                    fobj.write('dcl/dx %.15f \n'%(funcsSens[name+'_cl']['shape'][0][i]))




