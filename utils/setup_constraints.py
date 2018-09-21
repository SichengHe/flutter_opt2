# ======================================================================
#         DVConstraint Setup
# ====================================================================== 
DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)

# Only ADflow has the getTriangulatedSurface Function
DVCon.setSurface(CFDSolver.getTriangulatedMeshSurface())
le=0.02
leList = [[le    , 0, 0.0], [le    , 0, 1.0]]
teList = [[1.0-le, 0, 0.0], [1.0-le, 0, 1.0]]
# Thickness constraints
# DVCon.addThicknessConstraints2D(leList, teList, 2, 10, lower=0.85)

if comm.rank == 0:
    fileName = os.path.join(args.output, 'constraints.dat')
    # DVCon.writeTecplot(fileName)
