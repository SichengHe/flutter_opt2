# This is a template that should be used for setting up
# RANS analysis scripts

# ======================================================================
#         Import modules
# ======================================================================
import numpy
from mpi4py import MPI
from baseclasses import *
import argparse
import copy

# ======================================================================
#         Input Information -- Modify accordingly!
# ======================================================================


parser = argparse.ArgumentParser()
parser.add_argument("--flag_complex", help="bool var for complex to real implementation. True or False",
                   type=str, default=True)

args = parser.parse_args()

complex_flag = args.flag_complex
if complex_flag == 'False':
    complex_flag = False
else:
    complex_flag = True


if complex_flag:

    from adflow import ADFLOW_C
    # from pywarpustruct import USMesh_C

else:

    from adflow import ADFLOW
    # from pywarpustruct import USMesh 



outputDirectory = './OUTPUT'
gridFile = './INPUT/naca64A010_euler-L2.cgns'
areaRef = 1.0
chordRef = 1.0
MGCycle = 'sg'
name = 'fc'


alpha = 1.0000
mach = 0.73


if 1: 
    if complex_flag:
        # mach = complex(0.73, 1e-40)
        alpha = complex(alpha, 1e-40)
    else:
        pass


alpha_mag = 1.0/180*numpy.pi



omega = 80.0
# omega = 0.000000000001


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
    'MGCycle':MGCycle,
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

    'useblockettes' : False,
    }


if complex_flag:
    CFDSolver = ADFLOW_C(options=aeroOptions)
else:
    CFDSolver = ADFLOW(options=aeroOptions)

ap = AeroProblem(name=name, alpha=alpha, mach=mach, altitude=10000.0, areaRef=1.0, chordRef=1.0, evalFuncs=['cl','cd', 'cmz'],
    xRef=xRef,xRot=xRot,\
    degreePol=0,coefPol=[0.0],\
    degreeFourier=1,omegaFourier=omega,cosCoefFourier=[0.0,0.0],sinCoefFourier=[alpha_mag])


ap.addDV('mach',scale=1.0)
ap.addDV('alpha',scale=1.0)

funcs = {}
CFDSolver(ap)
CFDSolver.evalFunctions(ap, funcs)



funcsSens = {}
CFDSolver.evalFunctionsSens(ap, funcsSens)




if MPI.COMM_WORLD.rank == 0:

    if complex_flag:

        fobj=open(outputDirectory + '/' + 'obj.txt','w')
        fobj.write('fc_cl: %.15f, fc_cd:%.15f, fc_cmz:%.15f \n'%(numpy.imag(funcs['fc_cl']),numpy.imag(funcs['fc_cd']),numpy.imag(funcs['fc_cmz'])))

    if not complex_flag:

        fobj=open(outputDirectory + '/' + 'obj.txt','w')
        fobj.write('fc_cl: %.15f, fc_cd:%.15f, fc_cmz:%.15f \n'%(funcs['fc_cl'],funcs['fc_cd'],funcs['fc_cmz']))
        fobj.write('dfc_cl/dmach_fc:%.15f '%(funcsSens['fc_cl']['mach_fc']))
        fobj.write('dfc_cl/dalpha_fc:%.15f \n'%(funcsSens['fc_cl']['alpha_fc']))
        fobj.write('dfc_cd/dmach_fc:%.15f '%(funcsSens['fc_cd']['mach_fc']))
        fobj.write('dfc_cd/alpha_fc:%.15f \n'%(funcsSens['fc_cd']['alpha_fc']))
        fobj.write('dfc_cmz/dmach_fc:%.15f '%(funcsSens['fc_cmz']['mach_fc']))
        fobj.write('dfc_cmz/dalpha_fc:%.15f \n'%(funcsSens['fc_cmz']['alpha_fc']))

    fobj.close()
