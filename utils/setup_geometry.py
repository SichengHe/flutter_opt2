from pygeo import *
import numpy as np
from pyspline import *
# ======================================================================
#         DVGeometry Setup
# ====================================================================== 
# DVGeo = DVGeometry(FFDFile, complex=False)


DVGeo = DVGeometry(FFDFile, complex=True)

coef = DVGeo.FFD.vols[0].coef.copy()
coef_top, coef_bottom = map(np.array, zip(*coef))
coef_new = np.concatenate((coef_top,np.flipud(coef_bottom)), axis=0)
coef = coef_new


nSpan = coef.shape[0]
ref = np.zeros((nSpan*2,3))

for k in xrange(nSpan):
    ref[k,0] = np.average(coef[k,:,0])
    ref[k,1] = np.average(coef[k,:,1])
    ref[k,2] = 0.0

    ref[k + nSpan,0] = np.average(coef[k,:,0])
    ref[k + nSpan,1] = np.average(coef[k,:,1])
    ref[k + nSpan,2] = 1.0


X = ref
c0 = pySpline.Curve(X=X, k=2)
DVGeo.addRefAxis('axis', c0)




DVGeo.addGeoDVLocal('shape', lower=-1.0, upper=1.0, axis='y', scale=1.0)
    

