#!/usr/bin/python

import radia as rad
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def compute_fields(points_list, radia_object_id = None, field_component = 'bz'):

    m = len(points_list)
    k = m / size

    ## decompose the domain of the field calcaulation to the different processors and send the arrays
    if rank == 0:
        radia_object = rad.UtiDmp([radia_object_id], 'bin')

        for i in range(1, size):
            comm.send(radia_object, dest = i)

    else:
        radia_object = comm.recv(source = 0)

    device_id = rad.UtiDmpPrs(radia_object)

    ## convert the arrrays back into radia friendly format
    calc_points = np.asarray(points_list[rank::size])
    radia_points = np.ndarray.tolist(calc_points)

    # Compute the fields at the points specified 
    field_result = rad.Fld(device_id, field_component, radia_points)

    # construct arrays to gather points and field solution for plotting and analysis
    sendbuf = np.column_stack([calc_points, np.asarray(field_result)])
    row,col = sendbuf.shape

    recvbuf = None

    if rank == 0:
        recvbuf = np.empty([size, k+1, col])

    # Gather the arrays from the different processors
    comm.Gather(sendbuf, recvbuf, root = 0)

    # Return the field result
    if rank == 0:
        # clean up the data after collecting it
        data_out = recvbuf.reshape([size * (k + 1), col])
        bad_points = np.where((data_out > 1.0e20) + (data_out < -1.0e20) )

        data_out = np.delete(data_out, bad_points[0], axis = 0)

        # sort the data before returning, sort by x, then y, then z
        indices = np.lexsort((data_out[:,0], data_out[:,1], data_out[:,2]))
        data_out = [data_out[i,:]for i in indices]

        return np.asarray(data_out)

    else:
        return None




