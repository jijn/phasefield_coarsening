Parallel phase-field code for coarsening

Reference of the phase-field model:
Moelans, N. A quantitative and thermodynamically consistent phase-field interpolation function for multi-phase systems Acta Materialia, 2011, 59, 1077-1086

====================================================================

This code requires the OpenMPI library and the HDF5 library.


====================================================================

Compile the code:
icc -O2 -o ppf -L{dir of hdf5 library} -L{dir of mpi library} -lmpi -lhdf5 phasefield.c

If you run the code on the same machine the code is compiled, add compiler flag '-xhost' to enable more optimization.

Run the code:
mpirun -np N ./ppf filename 

N - number of cores used
filename - the input file name

====================================================================

The input hdf5 file can be generated with MATLAB:

etaS = ; etaL = ; c =; size = size(etaS);
h5create(filename,'/etaS',size,'Datatype','double');
h5create(filename,'/etaL',size,'Datatype','double');
h5create(filename,'/c',size,'Datatype','double');
h5write(filename,'/etaS',etaS);
h5write(filename,'/etaL',etaL);
h5write(filename,'/c',c);

