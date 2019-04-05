#!/usr/bin/python

import radia as rad
import uti_plot
import numpy
import time
import matplotlib.pyplot as plt
from mpi4py import MPI
import parallel_toolbox


def Und(lp, mp, np, cp, lm, mm, nm, cm, gap, gapOffset, numPer):

    zer = [0,0,0]
    Grp = rad.ObjCnt([])

    #Principal Poles and Magnets
    y = lp[1]/4;
    Pole = rad.ObjFullMag([lp[0]/4,y,-lp[2]/2-gap/2], [lp[0]/2,lp[1]/2,lp[2]], zer, np, Grp, mp, cp)
    y += lp[1]/4;

    mDir = -1
    for i in range(0, numPer):
        initM = [0, mDir, 0]; mDir *= -1
        y += lm[1]/2
        Magnet = rad.ObjFullMag([lm[0]/4,y,-lm[2]/2-gap/2-gapOffset], [lm[0]/2,lm[1],lm[2]], initM, nm, Grp, mm, cm)
        y += (lm[1] + lp[1])/2
        Pole = rad.ObjFullMag([lp[0]/4,y,-lp[2]/2-gap/2], [lp[0]/2,lp[1],lp[2]], zer, np, Grp, mp, cp)
        y += lp[1]/2

    initM = [0, mDir, 0]
    y += lm[1]/4;
    Magnet = rad.ObjFullMag([lm[0]/4,y,-lm[2]/2-gap/2-gapOffset], [lm[0]/2,lm[1]/2,lm[2]], initM, nm, Grp, mm, cm)

    #Mirrors
    rad.TrfZerPerp(Grp, [0,0,0], [1,0,0])
    rad.TrfZerPara(Grp, zer, [0,0,1])
    rad.TrfZerPerp(Grp, zer, [0,1,0])
    
    return Grp, Pole, Magnet

def Materials():
    #Defines magnetic materials for the Undulators Poles and Magnets
    #Pole (~iron type Va Permendur) material data
    H = [0.8, 1.5, 2.2, 3.6, 5, 6.8, 9.8, 18, 28, 37.5, 42, 55, 71.5, 80, 85, 88, 92, 100, 120, 150, 200, 300, 400, 600, 800, 1000, 2000, 4000, 6000, 10000, 25000, 40000]
    M = [0.000998995, 0.00199812, 0.00299724, 0.00499548,0.00699372, 0.00999145, 0.0149877, 0.0299774, 0.0499648, 0.0799529, 0.0999472, 0.199931, 0.49991, 0.799899, 0.999893, 1.09989, 1.19988, 1.29987, 1.41985, 1.49981, 1.59975, 1.72962, 1.7995, 1.89925, 1.96899, 1.99874,  2.09749, 2.19497, 2.24246, 2.27743, 2.28958, 2.28973]
    convH = 4.*3.141592653589793e-07
    ma = []
    for i in range(len(H)): ma.append([H[i]*convH, M[i]])
    mp = rad.MatSatIsoTab(ma)

    #(Permanent) Magnet material: NdFeB with 1.2 Tesla Remanent Magnetization
    mm = rad.MatStd('NdFeB', 1.2)

    return mp, mm


def GetMagnMaterCompMvsH(MeshH, ind, cmpnH,  cmpnM):
    #Extracting Magnetization vs Field Strength magnetic material data
    hMin = MeshH[0]; hMax = MeshH[1]; nh = MeshH[2]
    hStep = (hMax - hMin)/(nh - 1)
    M = [0]*nh
    sCmpnM = 'm' + cmpnM
    h = hMin
    H = [0,0,0]
    for i in range(nh):
        if(cmpnH == 'x'): H[0] = h
        elif(cmpnH == 'y'): H[1] = h
        elif(cmpnH == 'z'): H[2] = h
        M[i] = rad.MatMvsH(ind, sCmpnM, H)
        h += hStep
    return M


if __name__=="__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

	#General Undulator Parameters
    gap = 20; numPer = 50; per = 46; gapOffset = 1

    if rank == 0:

        print('RADIA Python Example #3:')
        print('This example creates and solves a simple U46 Hybrid Undulator made with rectangular magnet blocks.')
        print('')


        #Pole Parameters
        lp = [45,5,25]; np = [2,2,5]; cp = [1,0,1]
        ll = per/2 - lp[1]

        #Magnet Parameters
        lm = [65,ll,45]; nm = [1,3,1]; cm = [0,1,1]

        #Magnetic Materials
        mp, mm = Materials()
        tm1 = time.time()
        #Build the Structure
        und, pole, magnet = Und(lp, mp, np, cp, lm, mm, nm, cm, gap, gapOffset, numPer)
        tm2 = time.time()

        #Solve the Magnetization Problem, this is done on every processor at the moment
        res = rad.Solve(und, 0.0001, 1000)
        
        print('Relaxation Results:', res)
        print('Peak Magnetic Field:', rad.Fld(und, 'bz', [0,0,0]), 5, 'T')

    # Determine the post-processing domain for the field calculations and assemble the points list 
    yMax = per*(numPer+1)/2.
    yMin = -yMax
    ny = 5002
    yStep = (yMax - yMin)/(ny)
    xc = 0
    zc = 0
    y = numpy.arange(yMin, yMin + (ny) * yStep, yStep)

    points_list = []
    for i in range(ny):
        points_list.append([xc, y[i], zc])
    
    if rank > 0:
        und = None

    field_solution = parallel_toolbox.compute_fields(points_list, und, 'bz')

    if rank == 0:
        numpy.save('solution.npy', field_solution)

        #Plot the Results
        plt.figure(figsize = (15,5))
        plt.plot(field_solution[:,1], field_solution[:,3])
        plt.xlabel('y position [mm]')
        plt.ylabel('bz [T]')
        plt.show()

