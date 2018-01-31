/* phase-field code for coarsening
   
   reference:
   Moelans, N. A quantitative and thermodynamically consistent phase-field interpolation function
   for multi-phase systems Acta Materialia, 2011, 59, 1077-1086
   
   written by: Jin Zhang at DTU Physics
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define USE_MPI 1
#define USE_CACHE_BLOCKING 0
#define USE_BC_TYPE 1           /* 0: periodic BC */
                                /* 1: non-flux BC */

#if USE_MPI
#include <mpi.h>                /* MPI header file */
#endif

#include <hdf5.h>                /* use parallel IO (HDF5) */

/* define these two because Nieflheim use stderr */
//#define STD_BUF stdout            /* standard output */
#define STD_BUF stderr            /* standard output */
#define ERR_BUF stderr            /* standard error */

/* ----- domain size -----
   Notice here the coordinate system and the index space are not the same
   NQ^DIMENSION is the total number of processes
*/
/* domain size */
#if USE_MPI
#define NNX 300
#define NNY 300
#define NNZ 300
/* number of divides along each axis */
#define NQ 2
/* block size of each subdomain*/
#define NX NNX/NQ
#define NY NNY/NQ
#define NZ NNZ/NQ
#define NNI NNZ
#define NNJ NNY
#define NNK NNX
#else  /* not using MPI */
#define NX 100
#define NY 100
#define NZ 100
#endif    /* MPI */

/* map real space (x,y,z) to the array index (i,j,k) */
#define NI NZ
#define NJ NY
#define NK NX

#if USE_CACHE_BLOCKING
#define TJ NJ
#define TK NK
#define MIN(x,y) (x < y ? x : y)
#endif    /* Cache blocking */


/* define field variables (according to DIMENSION) */
/* move the field variables here to prevent stack problem */
double etaS[NI+2][NJ+2][NK+2],detaS[NI+2][NJ+2][NK+2];
double etaL[NI+2][NJ+2][NK+2],detaL[NI+2][NJ+2][NK+2];
double    c[NI+2][NJ+2][NK+2],   dc[NI+2][NJ+2][NK+2];
double   hS[NI+2][NJ+2][NK+2],   mu[NI+2][NJ+2][NK+2],M[NI+2][NJ+2][NK+2];

/* Usage: ppf filename */
int main(int argc, char **argv){
    /* -------------------------------------------------------------------- */
    /* material parameters */
    const double AS =  1e+09;
    const double AL =  1e+09;
    const double CS = -1e+06;
    const double CL =  1.e+07;
    const double cS0 = 0.455;
    const double cL0 = 0.978;
    const double cSeq = 0.999;
    const double cLeq = 0.476;
    const double DL = 1.0e-14, DS = 0.0;//6.0e-14;
    const double ML = DL/AL, MS = DS/AS;
    //const double lL = 7.6482e-09;
    const double sigma = 2.0;
    const double gamma = 1.5;
    /* mesh */
    const double dx = 1.440000000000000e-06;
    const double h2 = 1.0/(dx*dx);    /* 1/(dx)^2 */
    const double dt = 0.125*dx*dx/DL; /* to make the iteration stable */
    /* phase-field parameters */
    const double l = 7.0*dx;
    const double m = 6.0*sigma/l;
    const double kappa = 3*0.25*sigma*l;
    const double L = 4.0/3.0*m/kappa*(0.5*(MS+ML)/((cSeq-cLeq)*(cSeq-cLeq)));
    /* loop control variables */
    const double threshold = 1.0e-12;
    double error;
#if USE_MPI
    double error_local;
#endif
    const long n_iter=5000001L,n_save=50000L;//
    const long n_iter_init = 100000000L;
    long i_iter;
    char filename[50];

    /* other */
    int i,j,k;
    int ii,jj,kk;
    double tmp1,tmp2,tmp3,tmp4;
    double fS,fL,cS,cL,dmu;
    int save_iter;

    /* read the input filename */
    if( argc == 2 ){
        strcpy(filename,argv[1]);
        filename[strlen(filename)-3]='_';
        filename[strlen(filename)-2]='\0';
        fprintf(STD_BUF,"The job name is %s\n", filename);
    }else{
        fprintf(STD_BUF,"Usage:\nppf filename\n");
        return -1;
    }

#if USE_MPI
    /* MPI variables */
    char str_rank[5];              /* string stores the rank (for output) */
    MPI_Init(&argc, &argv);        /* start MPI */
    int rank,np;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;
    int ndims = 3;
    MPI_Comm topocomm;
    int coords[3];
    int rip,rim,rjp,rjm,rkp,rkm;
    MPI_Datatype norm_i_type, norm_j_type, norm_k_type;
    int ierror;
    init_mpi(comm,&rank,&np,ndims,&topocomm,coords,
             &rip,&rim,&rjp,&rjm,&rkp,&rkm,
             &norm_i_type,&norm_j_type,&norm_k_type);
    MPI_Type_commit(&norm_i_type);
    MPI_Type_commit(&norm_j_type);
    MPI_Type_commit(&norm_k_type);

    fprintf(STD_BUF,"[%d] num of processors: %d\n",rank,np);
    fprintf(STD_BUF,"[%d] (%d,%d,%d)\n",rank,coords[0],coords[1],coords[2]);
    fprintf(STD_BUF,"[%d] x{%d,%d} y{%d,%d} z{%d,%d}\n",rank,rim,rip,rjm,rjp,rkm,rkp);
#else
    int ndims = 3;
#endif    /* MPI */

    /* -------------------------------------------------------------------- */
    /* initialization - allocate memory*/
    for (i=0;i<NI+2;i++){
        for (j=0;j<NJ+2;j++){
            for (k=0;k<NK+2;k++){
                etaS[i][j][k] = 1.0; detaS[i][j][k] = 0.0;
                etaL[i][j][k] = 1.0; detaL[i][j][k] = 0.0;
                c   [i][j][k] = 1.0; dc   [i][j][k] = 0.0;
                hS  [i][j][k] = 1.0;
                mu  [i][j][k] = 1.0;
                M   [i][j][k] = 1.0;
            }
        }
    }
/* Note argv[1] saves the filename */
#if USE_MPI
    /* read initial value from h5 file (parallel IO) */
    read_hdf5_parallel(argv[1], topocomm, info, ndims, coords, etaS, etaL, c);
#else  /* not using MPI */
    read_hdf5(argv[1],ndims,etaS, etaL, c);
#endif    /* MPI */

    /* -------------------------------------------------------------------- */
    /* main loop */
    i_iter = 0L; error = threshold; save_iter = 0;
    while ( (i_iter<n_iter) &&(error>=threshold) ){
        
#if USE_MPI
        /* make sure all processes are ready for the new step*/
        MPI_Barrier(topocomm);
#endif    /* MPI */
        /* apply periodic boundary condition */
#if USE_MPI
        /* ghost-layer communication: send-receive - use vector type*/
#if USE_BC_TYPE == 0    /* periodic B.C. */
        if (i_iter<n_iter_init+1){ /* for the [initial] steps */
        exchange_ghost_layer(etaS,topocomm,rip,rim,rjp,rjm,rkp,rkm,&norm_i_type,&norm_j_type,&norm_k_type);
        exchange_ghost_layer(etaL,topocomm,rip,rim,rjp,rjm,rkp,rkm,&norm_i_type,&norm_j_type,&norm_k_type);
        }
        exchange_ghost_layer(   c,topocomm,rip,rim,rjp,rjm,rkp,rkm,&norm_i_type,&norm_j_type,&norm_k_type);
        /* ghost-layer communication: send-receive - version 2*/
        /*
        exchange_ghost_layer_v2(etaS,topocomm,coords,rip,rim,rjp,rjm,rkp,rkm,&norm_i_type,&norm_j_type,&norm_k_type);
        exchange_ghost_layer_v2(etaL,topocomm,coords,rip,rim,rjp,rjm,rkp,rkm,&norm_i_type,&norm_j_type,&norm_k_type);
        exchange_ghost_layer_v2(   c,topocomm,coords,rip,rim,rjp,rjm,rkp,rkm,&norm_i_type,&norm_j_type,&norm_k_type);
        */
#elif USE_BC_TYPE == 1          /* non-flux B.C. */
        exchange_ghost_layer(etaS,topocomm,rip,rim,rjp,rjm,rkp,rkm,&norm_i_type,&norm_j_type,&norm_k_type);
        exchange_ghost_layer(etaL,topocomm,rip,rim,rjp,rjm,rkp,rkm,&norm_i_type,&norm_j_type,&norm_k_type);
        exchange_ghost_layer(   c,topocomm,rip,rim,rjp,rjm,rkp,rkm,&norm_i_type,&norm_j_type,&norm_k_type);
        apply_nonflux_boundary_condition_mpi(etaS,etaL,c,topocomm,coords);
#endif
#else  /* not using MPI */
#if USE_BC_TYPE == 0             /* periodic B.C. */
        apply_periodic_boundary_condition(etaS,etaL,c);
#elif USE_BC_TYPE == 1           /* non-flux B.C. */
        apply_nonflux_boundary_condition(etaS,etaL,c);
#endif    /* USE_BC_TYPE */
#endif    /* MPI */

        if (i_iter<n_iter_init){ /* for the [initial] steps */
            sweep_fields(etaS,etaL,c,detaS,detaL,dc,hS,mu,M,MS,ML,AS,AL,cL0,cS0,CS,CL,L,dt,h2,m,gamma,kappa);
        }else{                    /* otherwise, only update c */
            sweep_fields_c(etaS,etaL,c,dc,hS,mu,M,MS,ML,AS,AL,cL0,cS0,CS,CL,dt,h2);
        }

        /* save results to file */
        if ( (i_iter>0) && ((((i_iter)%n_save)==0) || (i_iter==n_iter_init)) ){
#if USE_MPI
            //MPI_Barrier(topocomm);
#endif    /* MPI */
            /* write restart file (check point) */
            save_iter++;
            /* calculate the error*/
            error = 0.0;
            double error2=0.0,error3=0.0,error4=0.0;
            for (i=1;i<=NI;i++){
                for (j=1;j<=NJ;j++){
                    for (k=1;k<=NK;k++){
                        error += dc[i][j][k]*dc[i][j][k];
                        error3 += detaS[i][j][k]*detaS[i][j][k];
                        error4 += detaL[i][j][k]*detaL[i][j][k];
                    }
                }
            }
            error2 = error;
            fprintf(STD_BUF,"%ld: c=%10.4e; S=%10.4e; L=%10.4e\t",i_iter,error2,error3,error4);
#if USE_MPI
            error_local = error;
            error = 0.0;
            /* reduction error from all processors */
            MPI_Allreduce(&error_local, &error, 1, MPI_DOUBLE, MPI_SUM, topocomm);
#endif    /* MPI */
            error = sqrt(error);
            fprintf(STD_BUF,"iter: %ld; err: %10.4e\n",i_iter,error);
            /* write data to file */
#if USE_MPI
            write_result_file(filename,save_iter,ndims,topocomm,info,coords,etaS,etaL,c);
#else
            write_result_file(filename,save_iter,ndims,etaS,etaL,c);
#endif

        }    /* save results */
        i_iter++;
    } /* end of main loop */

#if USE_MPI
    MPI_Type_free(&norm_i_type); /* free user data types */
    MPI_Type_free(&norm_j_type);
    MPI_Type_free(&norm_k_type);
    MPI_Finalize();                /* CLOSE MPI */
#endif                            /* MPI */

    fprintf(STD_BUF,"Total steps: %ld; error: %10.6e\n",i_iter,error);
    return 0;
} /* main */

/* reverse:  reverse string s in place */
void reverse(char s[])
{
    int i, j;
    char c;
 
    for (i = 0, j = strlen(s)-1; i<j; i++, j--) {
        c = s[i];
        s[i] = s[j];
        s[j] = c;
    }
}

/* itoa:  convert n to characters in s */
void itoa(int n, char s[])
{
    int i, sign;
 
    if ((sign = n) < 0)  /* record sign */
        n = -n;          /* make n positive */
    i = 0;
    do {       /* generate digits in reverse order */
        s[i++] = n % 10 + '0';   /* get next digit */
    } while ((n /= 10) > 0);     /* delete it */
    if (sign < 0)
        s[i++] = '-';
    s[i] = '\0';
    reverse(s);
}

int sweep_fields(double etaS[NI+2][NJ+2][NK+2],
                 double etaL[NI+2][NJ+2][NK+2],
                 double c[NI+2][NJ+2][NK+2],
                 double detaS[NI+2][NJ+2][NK+2],
                 double detaL[NI+2][NJ+2][NK+2],
                 double dc[NI+2][NJ+2][NK+2],
                 double hS[NI+2][NJ+2][NK+2],
                 double mu[NI+2][NJ+2][NK+2],
                 double M[NI+2][NJ+2][NK+2],
                 const double MS , const double ML ,
                 const double AS , const double AL ,
                 const double cL0, const double cS0,
                 const double CS , const double CL ,
                 const double L, const double dt, const double h2,
                 const double m, const double gamma, const double kappa){
    double tmp4;
    double cS,cL,fS,fL,dmu;
    int i,j,k;
#if USE_CACHE_BLOCKING
    int jj,kk;
#endif
    const double register tmp5 = AS*AL;
    const double register tmp6 = cL0-cS0;
    /* calculate temporary variables (w/ ghost layer)*/
    for (i=0;i<NI+2;i++){
        for (j=0;j<NJ+2;j++){
            for (k=0;k<NK+2;k++){
                hS[i][j][k] = etaS[i][j][k]*etaS[i][j][k]/
                             (etaS[i][j][k]*etaS[i][j][k]+
                              etaL[i][j][k]*etaL[i][j][k]);
                mu[i][j][k] = tmp5/(AS+(AL-AS)*hS[i][j][k])*(c[i][j][k]-cL0+tmp6*hS[i][j][k]);
                M [i][j][k] = ML+(MS-ML)*hS[i][j][k];
            }
        }
    }
    /* sweep field variables */
    const double register tmp1 = -1.0*L*dt;
    const double register tmp2 = (AS*cS0-AL*cL0);
    const double register tmp3 = dt*0.5*h2;
#if USE_CACHE_BLOCKING
    for (jj=1;jj<=NJ;jj+=TJ){
        for (kk=1;kk<=NK;kk+=TK){
            for (i=1;i<=NI;i++){
                for (j=jj;j<=MIN(jj+TJ-1,NJ);j++){
                    for (k=kk;k<=MIN(kk+TK-1,NK);k++){
#else
    for (i=1;i<=NI;i++){
        for (j=1;j<=NJ;j++){
            for (k=1;k<=NK;k++){
#endif    /* cache blocking */
                tmp4 = 1.0/(AS+(AL-AS)*hS[i][j][k]);
                cS = tmp4*(AL*c[i][j][k]+(1.0-hS[i][j][k])*tmp2);
                cL = tmp4*(AS*c[i][j][k]-     hS[i][j][k] *tmp2);
                fS = 0.5*AS*(cS-cS0)*(cS-cS0)+CS;
                fL = 0.5*AL*(cL-cL0)*(cL-cL0)+CL;
                dmu = 2.0*(etaS[i][j][k]*etaL[i][j][k])/(
                                                         (etaS[i][j][k]*etaS[i][j][k] + etaL[i][j][k]*etaL[i][j][k])*
                                                         (etaS[i][j][k]*etaS[i][j][k] + etaL[i][j][k]*etaL[i][j][k])
                                                        )*(fS-fL-mu[i][j][k]*(cS-cL));
                detaS[i][j][k] = tmp1*(
                                       m*etaS[i][j][k]*(etaS[i][j][k]*etaS[i][j][k]-1.0+2.0*gamma*etaL[i][j][k]*etaL[i][j][k])
                                       -kappa*h2*(etaS[i+1][j][k]+etaS[i-1][j][k]+
                                                     etaS[i][j+1][k]+etaS[i][j-1][k]+
                                                     etaS[i][j][k+1]+etaS[i][j][k-1]-6.0*etaS[i][j][k])
                                       +etaL[i][j][k]*dmu
                                      );
                detaL[i][j][k] = tmp1*(
                                       m*etaL[i][j][k]*(etaL[i][j][k]*etaL[i][j][k]-1.0+2.0*gamma*etaS[i][j][k]*etaS[i][j][k])
                                       -kappa*h2*(etaL[i+1][j][k]+etaL[i-1][j][k]+
                                                  etaL[i][j+1][k]+etaL[i][j-1][k]+
                                                  etaL[i][j][k+1]+etaL[i][j][k-1]-6.0*etaL[i][j][k])
                                       -etaS[i][j][k]*dmu
                                      );
                dc[i][j][k] = tmp3*(
                                    M[i+1][j  ][k  ]*(mu[i+1][j  ][k  ]-mu[i  ][j  ][k  ]) -
                                    M[i-1][j  ][k  ]*(mu[i  ][j  ][k  ]-mu[i-1][j  ][k  ]) +
                                    M[i  ][j+1][k  ]*(mu[i  ][j+1][k  ]-mu[i  ][j  ][k  ]) -
                                    M[i  ][j-1][k  ]*(mu[i  ][j  ][k  ]-mu[i  ][j-1][k  ]) +
                                    M[i  ][j  ][k+1]*(mu[i  ][j  ][k+1]-mu[i  ][j  ][k  ]) -
                                    M[i  ][j  ][k-1]*(mu[i  ][j  ][k  ]-mu[i  ][j  ][k-1]) +
                                    M[i  ][j  ][k  ]*(mu[i+1][j  ][k  ]+mu[i-1][j  ][k  ]+
                                                      mu[i  ][j+1][k  ]+mu[i  ][j-1][k  ]+
                                                      mu[i  ][j  ][k+1]+mu[i  ][j  ][k-1]-6.0*mu[i][j][k])
                                   );
#if USE_CACHE_BLOCKING
                    }
                }
#endif    /* cache blocking */
            }
        }
    }
    /* add the increment - contiguous memory accesss - fast*/
    for (i=0;i<NI+2;i++){
        for (j=0;j<NJ+2;j++){
            for (k=0;k<NK+2;k++){
                etaS[i][j][k] = etaS[i][j][k] + detaS[i][j][k];
                etaL[i][j][k] = etaL[i][j][k] + detaL[i][j][k];
                c   [i][j][k] = c   [i][j][k] + dc   [i][j][k];
            }
        }
    }
    return 0;
};


int sweep_fields_c(double etaS[NI+2][NJ+2][NK+2],
                   double etaL[NI+2][NJ+2][NK+2],
                   double c [NI+2][NJ+2][NK+2],
                   double dc[NI+2][NJ+2][NK+2],
                   double hS[NI+2][NJ+2][NK+2],
                   double mu[NI+2][NJ+2][NK+2],
                   double M [NI+2][NJ+2][NK+2],
                   const double MS , const double ML ,
                   const double AS , const double AL ,
                   const double cL0, const double cS0,
                   const double CS , const double CL ,
                   const double dt, const double h2){
    int i,j,k;
#if USE_CACHE_BLOCKING
    int jj,kk;
#endif
    const double register tmp5 = AS*AL;
    const double register tmp6 = cL0-cS0;
    /* calculate temporary variables (w/ ghost layer)*/
    for (i=0;i<NI+2;i++){
        for (j=0;j<NJ+2;j++){
            for (k=0;k<NK+2;k++){
                hS[i][j][k] = etaS[i][j][k]*etaS[i][j][k]/
                             (etaS[i][j][k]*etaS[i][j][k]+
                              etaL[i][j][k]*etaL[i][j][k]);
                mu[i][j][k] = tmp5/(AS+(AL-AS)*hS[i][j][k])*(c[i][j][k]-cL0+tmp6*hS[i][j][k]);
                M [i][j][k] = ML+(MS-ML)*hS[i][j][k];
            }
        }
    }
    /* sweep field variables */
    const double register tmp3 = dt*0.5*h2;
#if USE_CACHE_BLOCKING
    for (jj=1;jj<=NJ;jj+=TJ){
        for (kk=1;kk<=NK;kk+=TK){
            for (i=1;i<=NI;i++){
                for (j=jj;j<=MIN(jj+TJ-1,NJ);j++){
                    for (k=kk;k<=MIN(kk+TK-1,NK);k++){
#else
    for (i=1;i<=NI;i++){
        for (j=1;j<=NJ;j++){
            for (k=1;k<=NK;k++){
#endif    /* cache blocking */
                dc[i][j][k] = tmp3*(
                                    M[i+1][j  ][k  ]*(mu[i+1][j  ][k  ]-mu[i  ][j  ][k  ]) -
                                    M[i-1][j  ][k  ]*(mu[i  ][j  ][k  ]-mu[i-1][j  ][k  ]) +
                                    M[i  ][j+1][k  ]*(mu[i  ][j+1][k  ]-mu[i  ][j  ][k  ]) -
                                    M[i  ][j-1][k  ]*(mu[i  ][j  ][k  ]-mu[i  ][j-1][k  ]) +
                                    M[i  ][j  ][k+1]*(mu[i  ][j  ][k+1]-mu[i  ][j  ][k  ]) -
                                    M[i  ][j  ][k-1]*(mu[i  ][j  ][k  ]-mu[i  ][j  ][k-1]) +
                                    M[i  ][j  ][k  ]*(mu[i+1][j  ][k  ]+mu[i-1][j  ][k  ]+
                                                      mu[i  ][j+1][k  ]+mu[i  ][j-1][k  ]+
                                                      mu[i  ][j  ][k+1]+mu[i  ][j  ][k-1]-6.0*mu[i][j][k])
                                   );
#if USE_CACHE_BLOCKING
                    }
                }
#endif    /* cache blocking */
            }
        }
    }
    /* add the increment - contiguous memory accesss - fast*/
    for (i=0;i<NI+2;i++){
        for (j=0;j<NJ+2;j++){
            for (k=0;k<NK+2;k++){
                c   [i][j][k] = c   [i][j][k] + dc   [i][j][k];
            }
        }
    }
    return 0;
}

            
#if USE_MPI
/* parallel read h5 file */
int read_hdf5_parallel(char filename[], MPI_Comm topocomm, MPI_Info info,
                       int ndims, int coords[3],
                       double etaS[NI+2][NJ+2][NK+2],
                       double etaL[NI+2][NJ+2][NK+2],
                       double c   [NI+2][NJ+2][NK+2]){

    /* HDF5 library variables */
    hid_t file_id,dataset_id_S,dataset_id_L,dataset_id_c;
    hid_t filespace_id,memspace_id;
    hid_t fapl_id;                /* file access property list */
    hid_t xfpl_id;                /* file transfer property list */
    hsize_t h5dims[3],offset[3], stride[3],count[3],block[3];
    herr_t status;
    
    /* Set up file access property list with parallel IO */
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, topocomm, info);
    /* open the input file with parallel I/O access properties */
    file_id = H5Fopen(filename, H5F_ACC_RDONLY,fapl_id);
    status = H5Pclose(fapl_id);
    /* Open existing datasets */
    dataset_id_S = H5Dopen2(file_id, "/etaS",H5P_DEFAULT);
    /* create a memory dataspace and set the hyperslab */
    h5dims[0] = NI+2; h5dims[1] = NJ+2; h5dims[2] = NK+2;
    stride[0] = 1   ; stride[1] = 1   ; stride[2] = 1   ;
    count [0] = NI  ; count [1] = NJ  ; count [2] = NK  ;
    block [0] = 1   ; block [1] = 1   ; block [2] = 1   ;
    memspace_id = H5Screate_simple(ndims,h5dims,NULL);
    offset[0] = 1; offset[1] = 1; offset[2] = 1;
    status = H5Sselect_hyperslab(memspace_id,H5S_SELECT_SET,offset,stride,count,block);
    /* get the file dataspace and set the hyperslab */
    offset[0] = coords[0]*NI; offset[1] = coords[1]*NJ; offset[2] = coords[2]*NK;
    filespace_id = H5Dget_space(dataset_id_S);
    status = H5Sselect_hyperslab(filespace_id,H5S_SELECT_SET,offset,stride,count,block);

    /* create property list for collective dataset read */
    xfpl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfpl_id, H5FD_MPIO_COLLECTIVE);
    
    /* read a subset to etaS */
    H5Dread(dataset_id_S,H5T_IEEE_F64LE,memspace_id,filespace_id,xfpl_id,etaS);
    status = H5Sclose(filespace_id);
    status = H5Dclose(dataset_id_S);
    
    /* read a subset to etaL */
    dataset_id_L = H5Dopen2(file_id, "/etaL",H5P_DEFAULT);
    filespace_id = H5Dget_space(dataset_id_L);
    status = H5Sselect_hyperslab(filespace_id,H5S_SELECT_SET,offset,stride,count,block);
    H5Dread(dataset_id_S,H5T_IEEE_F64LE,memspace_id,filespace_id,xfpl_id,etaL);
    status = H5Sclose(filespace_id);
    status = H5Dclose(dataset_id_L);
    
    /* read a subset to c */
    dataset_id_c = H5Dopen2(file_id,    "/c",H5P_DEFAULT);
    filespace_id = H5Dget_space(dataset_id_c);
    status = H5Sselect_hyperslab(filespace_id,H5S_SELECT_SET,offset,stride,count,block);
    H5Dread(dataset_id_c,H5T_IEEE_F64LE,memspace_id,filespace_id,xfpl_id,c);
    status = H5Sclose(filespace_id);
    status = H5Dclose(dataset_id_c);

    /* close stuff */
    status = H5Pclose(xfpl_id);
    status = H5Sclose(memspace_id);
    status = H5Fclose(file_id);
    
    return 0;
}


int write_hdf5_parallel(char filename[], MPI_Comm topocomm, MPI_Info info,
                        int ndims, int coords[3],
                        double etaS[NI+2][NJ+2][NK+2],
                        double etaL[NI+2][NJ+2][NK+2],
                        double c   [NI+2][NJ+2][NK+2]){

    /* HDF5 library variables */
    hid_t file_id,dataset_id_S,dataset_id_L,dataset_id_c;
    hid_t filespace_id,memspace_id;
    hid_t fapl_id;                /* file access property list */
    hid_t xfpl_id;                /* file transfer property list */
    hid_t plist_id;
    hsize_t h5dims[3],offset[3], stride[3],count[3],block[3];
    hsize_t chunk_dims[3];
    herr_t status;
    /* Set up file access property list with parallel IO */
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_id, topocomm, info);
    /* creat the output file with default I/O access properties */
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC,H5P_DEFAULT,fapl_id);
    status = H5Pclose(fapl_id);
    /* create dataspaces */
    h5dims[0] = NI+2; h5dims[1] = NJ+2; h5dims[2] = NK+2;
    memspace_id = H5Screate_simple(ndims,h5dims,NULL);
    h5dims[0] = NNI; h5dims[1] = NNJ; h5dims[2] = NNK;
    filespace_id = H5Screate_simple(ndims,h5dims,NULL);
    /* create chunked dataset */
    chunk_dims[0] = NI; chunk_dims[1] = NJ; chunk_dims[2] = NK;
    plist_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id, ndims, chunk_dims);
    dataset_id_S = H5Dcreate2(file_id, "etaS", H5T_IEEE_F64LE, filespace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    dataset_id_L = H5Dcreate2(file_id, "etaL", H5T_IEEE_F64LE, filespace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    dataset_id_c = H5Dcreate2(file_id,    "c", H5T_IEEE_F64LE, filespace_id, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    H5Pclose(plist_id);
    H5Sclose(filespace_id);
    /* set hyperslab */
    stride[0] = 1   ; stride[1] = 1   ; stride[2] = 1   ;
    count [0] = NI  ; count [1] = NJ  ; count [2] = NK  ;
    block [0] = 1   ; block [1] = 1   ; block [2] = 1   ;
    offset[0] = 1   ; offset[1] = 1   ; offset[2] = 1   ;
    status = H5Sselect_hyperslab(memspace_id,H5S_SELECT_SET,offset,stride,count,block);
    offset[0] = coords[0]*NI; offset[1] = coords[1]*NJ; offset[2] = coords[2]*NK;
    filespace_id = H5Dget_space(dataset_id_S);
    status = H5Sselect_hyperslab(filespace_id,H5S_SELECT_SET,offset,stride,count,block);

    /* create property list for collective dataset read */
    xfpl_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xfpl_id, H5FD_MPIO_COLLECTIVE);

    /* write etaS to file*/
    H5Dwrite(dataset_id_S,H5T_IEEE_F64LE,memspace_id,filespace_id,xfpl_id,etaS);
    status = H5Sclose(filespace_id);
    status = H5Dclose(dataset_id_S);

    /* write etaL to file */
    filespace_id = H5Dget_space(dataset_id_L);
    status = H5Sselect_hyperslab(filespace_id,H5S_SELECT_SET,offset,stride,count,block);
    H5Dwrite(dataset_id_L,H5T_IEEE_F64LE,memspace_id,filespace_id,xfpl_id,etaL);
    status = H5Sclose(filespace_id);
    status = H5Dclose(dataset_id_L);
    

    /* write c to file */
    filespace_id = H5Dget_space(dataset_id_c);
    status = H5Sselect_hyperslab(filespace_id,H5S_SELECT_SET,offset,stride,count,block);
    H5Dwrite(dataset_id_c,H5T_IEEE_F64LE,memspace_id,filespace_id,xfpl_id,c);
    status = H5Sclose(filespace_id);
    status = H5Dclose(dataset_id_c);

    /* close other stuff */
    status = H5Pclose(xfpl_id);
    status = H5Sclose(memspace_id);
    status = H5Fclose(file_id);
    return 0;
}


int init_mpi(MPI_Comm comm, int *rank, int *np, int ndims,
             MPI_Comm *topocomm, int coords[3],
             int *rip, int *rim,
             int *rjp, int *rjm,
             int *rkp, int *rkm,
             MPI_Datatype *norm_i_type,
             MPI_Datatype *norm_j_type,
             MPI_Datatype *norm_k_type){

    MPI_Comm_rank(comm, rank);    /* get rank id */
    MPI_Comm_size(comm, np);    /* get number of processors */

    /* a simple check */
    if (NQ*NQ*NQ!=*np){
        fprintf(ERR_BUF,"[%d]NQ=%d is wrong\n",*rank,NQ);
        MPI_Abort(comm,1);
        return -1;
    }

    /* MPI compute domain decomposition */
    int pdims[3]={0,0,0};
    MPI_Dims_create(*np, ndims, pdims);
    
    /* MPI create Cartesian topology */
#if USE_BC_TYPE == 0             /* periodic B.C. */
    int periods[3] = {1,1,1};    /* periodic topology */
#elif USE_BC_TYPE == 1           /* non-flux B.C. */
    int periods[3] = {0,0,0};    /* non-periodic topology */
#else
    /* TODO: other boundary conditions */
#endif /* USE_BC_TYPE */
    MPI_Cart_create(comm, ndims, pdims, periods, 0, topocomm);

    /* MPI get coordinate */
    MPI_Comm_rank(*topocomm, rank); /* get rank id in topocomm*/
    MPI_Cart_coords(*topocomm, *rank, 3, coords);

    /* MPI get neighbor */
    MPI_Cart_shift(*topocomm, 0, 1, rim, rip);
    MPI_Cart_shift(*topocomm, 1, 1, rjm, rjp);
    MPI_Cart_shift(*topocomm, 2, 1, rkm, rkp);

    /* User data types */
    MPI_Datatype tmp_type;
    MPI_Type_vector(NJ, NK, NK+2, MPI_DOUBLE, norm_i_type); /* the plane (jk) normal to i direction */
    MPI_Type_vector(NI, NK, (NJ+2)*(NK+2), MPI_DOUBLE, norm_j_type); /* the plane (ik) normal to j direction */
    MPI_Type_vector(NJ, 1, NK+2, MPI_DOUBLE, &tmp_type);
    MPI_Type_commit(&tmp_type);
    MPI_Type_create_hvector(NI, 1, (NK+2)*(NJ+2)*sizeof(MPI_DOUBLE), tmp_type, norm_k_type);
    MPI_Type_free(&tmp_type);

    return 0;
}


/* ghost-layer communication: send-receive - use vector type*/
int exchange_ghost_layer(double u[NI+2][NJ+2][NK+2],
                         MPI_Comm topocomm,
                         int rip, int rim,
                         int rjp, int rjm,
                         int rkp, int rkm,
                         MPI_Datatype *norm_i_type,
                         MPI_Datatype *norm_j_type,
                         MPI_Datatype *norm_k_type ){
    MPI_Request reqs[12];
    MPI_Isend(&u[   1][1][1],1,*norm_i_type,rim,9,topocomm,&reqs[ 0]); /* send to i- */
    MPI_Irecv(&u[NI+1][1][1],1,*norm_i_type,rip,9,topocomm,&reqs[ 1]); /* receive from i+ */
    MPI_Isend(&u[  NI][1][1],1,*norm_i_type,rip,6,topocomm,&reqs[ 2]); /* send to i+ */
    MPI_Irecv(&u[   0][1][1],1,*norm_i_type,rim,6,topocomm,&reqs[ 3]); /* receive from i- */
        
    MPI_Isend(&u[1][   1][1],1,*norm_j_type,rjm,8,topocomm,&reqs[ 4]); /* send to j- */
    MPI_Irecv(&u[1][NJ+1][1],1,*norm_j_type,rjp,8,topocomm,&reqs[ 5]); /* receive from j+ */
    MPI_Isend(&u[1][  NJ][1],1,*norm_j_type,rjp,5,topocomm,&reqs[ 6]); /* send to j+ */
    MPI_Irecv(&u[1][   0][1],1,*norm_j_type,rjm,5,topocomm,&reqs[ 7]); /* receive from j- */
        
    MPI_Isend(&u[1][1][   1],1,*norm_k_type,rkm,7,topocomm,&reqs[ 8]); /* send to k- */
    MPI_Irecv(&u[1][1][NK+1],1,*norm_k_type,rkp,7,topocomm,&reqs[ 9]); /* receive from k+ */
    MPI_Isend(&u[1][1][  NK],1,*norm_k_type,rkp,4,topocomm,&reqs[10]); /* send to k+ */
    MPI_Irecv(&u[1][1][   0],1,*norm_k_type,rkm,4,topocomm,&reqs[11]); /* receive from k- */
    MPI_Waitall(12, reqs, MPI_STATUSES_IGNORE);
    return 0;
}

/* ghost-layer communication: send-receive - use vector type
   version 2 */
int exchange_ghost_layer_v2(double u[NI+2][NJ+2][NK+2],
                            MPI_Comm topocomm, int coords[3],
                            int rip, int rim,
                            int rjp, int rjm,
                            int rkp, int rkm,
                            MPI_Datatype *norm_i_type,
                            MPI_Datatype *norm_j_type,
                            MPI_Datatype *norm_k_type ){
    MPI_Status status;
    /* send/recv the plane normal to i */
    if (0==coords[0]%2){
        MPI_Send(&u[   1][1][1],1,*norm_i_type,rim,9,topocomm); /* send to i- */
        MPI_Send(&u[  NI][1][1],1,*norm_i_type,rip,6,topocomm); /* send to i+ */
        MPI_Recv(&u[NI+1][1][1],1,*norm_i_type,rip,9,topocomm,&status); /* receive from i+ */
        MPI_Recv(&u[   0][1][1],1,*norm_i_type,rim,6,topocomm,&status); /* receive from i- */
    }else{
        MPI_Recv(&u[NI+1][1][1],1,*norm_i_type,rip,9,topocomm,&status); /* receive from i+ */
        MPI_Recv(&u[   0][1][1],1,*norm_i_type,rim,6,topocomm,&status); /* receive from i- */
        MPI_Send(&u[   1][1][1],1,*norm_i_type,rim,9,topocomm); /* send to i- */
        MPI_Send(&u[  NI][1][1],1,*norm_i_type,rip,6,topocomm); /* send to i+ */
    }
    /* send/recv the plane normal to j */
    if (0==coords[1]%2){
        MPI_Send(&u[1][   1][1],1,*norm_j_type,rjm,8,topocomm); /* send to j- */
        MPI_Send(&u[1][  NJ][1],1,*norm_j_type,rjp,5,topocomm); /* send to j+ */
        MPI_Recv(&u[1][NJ+1][1],1,*norm_j_type,rjp,8,topocomm,&status); /* receive from j+ */
        MPI_Recv(&u[1][   0][1],1,*norm_j_type,rjm,5,topocomm,&status); /* receive from j- */
    }else{
        MPI_Recv(&u[1][NJ+1][1],1,*norm_j_type,rjp,8,topocomm,&status); /* receive from j+ */
        MPI_Recv(&u[1][   0][1],1,*norm_j_type,rjm,5,topocomm,&status); /* receive from j- */
        MPI_Send(&u[1][   1][1],1,*norm_j_type,rjm,8,topocomm); /* send to j- */
        MPI_Send(&u[1][  NJ][1],1,*norm_j_type,rjp,5,topocomm); /* send to j+ */
    }
    /* send/recv the plane normal to k */
    if (0==coords[2]%2){
        MPI_Send(&u[1][1][   1],1,*norm_k_type,rkm,7,topocomm); /* send to k- */
        MPI_Send(&u[1][1][  NK],1,*norm_k_type,rkp,4,topocomm); /* send to k+ */
        MPI_Recv(&u[1][1][NK+1],1,*norm_k_type,rkp,7,topocomm,&status); /* receive from k+ */
        MPI_Recv(&u[1][1][   0],1,*norm_k_type,rkm,4,topocomm,&status); /* receive from k- */
    }else{
        MPI_Recv(&u[1][1][NK+1],1,*norm_k_type,rkp,7,topocomm,&status); /* receive from k+ */
        MPI_Recv(&u[1][1][   0],1,*norm_k_type,rkm,4,topocomm,&status); /* receive from k- */
        MPI_Send(&u[1][1][   1],1,*norm_k_type,rkm,7,topocomm); /* send to k- */
        MPI_Send(&u[1][1][  NK],1,*norm_k_type,rkp,4,topocomm); /* send to k+ */
    }
    return 0;
}

/* after exchange ghost layer, apply the non-flux B.C. */
/* only consider cells at boundary */
int apply_nonflux_boundary_condition_mpi(double etaS[NI+2][NJ+2][NK+2],
                                         double etaL[NI+2][NJ+2][NK+2],
                                         double c   [NI+2][NJ+2][NK+2],
                                         MPI_Comm topocomm,int coords[3]){
    int i,j,k;
    /* if coords == 0 OR coords == NQ-1: deal with non-flux B.C. */
    /* direction i */
    if (0==coords[0]){          /* i- */
        for (j=1;j<=NJ;j++){
            for (k=1;k<=NK;k++){
                etaS[   0][j][k] = etaS[ 1][j][k];
                etaL[   0][j][k] = etaL[ 1][j][k];
                c   [   0][j][k] = c   [ 1][j][k];
            }
        }
    }
    if ((NQ-1)==coords[0]){     /* i+ */
        for (j=1;j<=NJ;j++){
            for (k=1;k<=NK;k++){
                etaS[NI+1][j][k] = etaS[NI][j][k];
                etaL[NI+1][j][k] = etaL[NI][j][k];
                c   [NI+1][j][k] = c   [NI][j][k];
            }
        }
    }
    /* direction j */
    if (0==coords[1]){          /* j- */
        for (i=1;i<=NI;i++){
            for (k=1;k<=NK;k++){
                etaS[i][   0][k] = etaS[i][ 1][k];
                etaL[i][   0][k] = etaL[i][ 1][k];
                c   [i][   0][k] = c   [i][ 1][k];
            }
        }
    }
    if ((NQ-1)==coords[1]){     /* j+ */
        for (i=1;i<=NI;i++){
            for (k=1;k<=NK;k++){
                etaS[i][NJ+1][k] = etaS[i][NJ][k];
                etaL[i][NJ+1][k] = etaL[i][NJ][k];
                c   [i][NJ+1][k] = c   [i][NJ][k];
            }
        }
    }
    /* direction k */
    if (0==coords[2]){          /* k- */
        for (i=1;i<=NI;i++){
            for (j=1;j<=NJ;j++){
                etaS[i][j][   0] = etaS[i][j][ 1];
                etaL[i][j][   0] = etaL[i][j][ 1];
                c   [i][j][   0] = c   [i][j][ 1];
            }
        }
    }
    if ((NQ-1)==coords[2]){     /* k+ */
        for (i=1;i<=NI;i++){
            for (j=1;j<=NJ;j++){
                etaS[i][j][NK+1] = etaS[i][j][NK];
                etaL[i][j][NK+1] = etaL[i][j][NK];
                c   [i][j][NK+1] = c   [i][j][NK];
            }
        }
    }
    return 0;
}

void handel_mpi_error(int error){
    switch(error){
    case 1:
        fprintf(ERR_BUF,"Invalid buffer pointer.\n");
        break;
    case 2:
        fprintf(ERR_BUF,"Invalid count argument.\n");
        break;
    case 3:
        fprintf(ERR_BUF,"Invalid datatype argument.\n");
        break;
    case 4:
        fprintf(ERR_BUF,"Invalid tag argument.\n");
        break;
    case 5:
        fprintf(ERR_BUF,"Invalid communicator.\n");
        break;
    case 6:
        fprintf(ERR_BUF,"Invalid rank.\n");
        break;
    case 7:
        fprintf(ERR_BUF,"Invalid MPI_Request handle.\nInvalid root.\n");
        break;
    case 8:
        fprintf(ERR_BUF," Null group passed to function.\n");
        break;
    case 9:
        fprintf(ERR_BUF,"Invalid operation.\n");
        break;
    case 10:
        fprintf(ERR_BUF,"Invalid topology.\n");
        break;
    case 11:
        fprintf(ERR_BUF,"Illegal dimension argument.\n");
        break;
    case 12:
        fprintf(ERR_BUF,"Invalid argument.\n");
        break;
    case 13:
        fprintf(ERR_BUF,"Unknown error.\n");
        break;
    case 14:
        fprintf(ERR_BUF,"Message truncated on receive.\n");
        break;
    case 15:
        fprintf(ERR_BUF,"Other error; use Error_string.\n");
        break;
    case 16:
        fprintf(ERR_BUF,"Internal error code.\n");
        break;
    case 17:
        fprintf(ERR_BUF,"Look in status for error value.\n");
        break;
    case 18:
        fprintf(ERR_BUF,"Pending request.\n");
        break;
    case 19:
        fprintf(ERR_BUF,"Permission denied.\n");
        break;
    case 20:
        fprintf(ERR_BUF,"Unsupported amode passed to open.\n");
        break;
    case 21:
        fprintf(ERR_BUF,"Invalid assert.\n");
        break;
    case 22:
        fprintf(ERR_BUF,"Invalid file name (for example, path name too long).\n");
        break;
    case 23:
        fprintf(ERR_BUF,"Invalid base.\n");
        break;
    case 24:
        fprintf(ERR_BUF,"An error occurred in a user-supplied data-conversion function.\n");
        break;
    case 25:
        fprintf(ERR_BUF,"Invalid displacement.\n");
        break;
    case 26:
        fprintf(ERR_BUF,"Conversion functions could not be registered because a data representation identifier that was already defined was passed to MPI_REGISTER_DATAREP.\n");
        break;
    case 27:
        fprintf(ERR_BUF,"File exists.\n");
        break;
    case 28:
        fprintf(ERR_BUF,"File operation could not be completed, as the file is currently open by some process.\n");
        break;
    case 29:
        fprintf(ERR_BUF,"MPI_ERR_FILE\n");
        break;
    case 30:
        fprintf(ERR_BUF,"Illegal info key.\n");
        break;
    case 31:
        fprintf(ERR_BUF,"No such key.\n");
        break;
    case 32:
        fprintf(ERR_BUF,"Illegal info value.\n");
        break;
    case 33:
        fprintf(ERR_BUF,"Invalid info object.\n");
        break;
    case 34:
        fprintf(ERR_BUF,"I/O error.\n");
        break;
    case 35:
        fprintf(ERR_BUF,"Illegal key value.\n");
        break;
    case 36:
        fprintf(ERR_BUF,"Invalid locktype.\n");
        break;
    case 37:
        fprintf(ERR_BUF,"Name not found.\n");
        break;
    case 38:
        fprintf(ERR_BUF,"Memory exhausted.\n");
        break;
    case 39:
        fprintf(ERR_BUF,"MPI_ERR_NOT_SAME\n");
        break;
    case 40:
        fprintf(ERR_BUF,"Not enough space.\n");
        break;
    case 41:
        fprintf(ERR_BUF,"File (or directory) does not exist.\n");
        break;
    case 42:
        fprintf(ERR_BUF,"Invalid port.\n");
        break;
    case 43:
        fprintf(ERR_BUF,"Quota exceeded.\n");
        break;
    case 44:
        fprintf(ERR_BUF,"Read-only file system.\n");
        break;
    case 45:
        fprintf(ERR_BUF,"Conflicting accesses to window.\n");
        break;
    case 46:
        fprintf(ERR_BUF,"Erroneous RMA synchronization.\n");
        break;
    case 47:
        fprintf(ERR_BUF,"Invalid publish/unpublish.\n");
        break;
    case 48:
        fprintf(ERR_BUF,"Invalid size.\n");
        break;
    case 49:
        fprintf(ERR_BUF,"Error spawning.\n");
        break;
    case 50:
        fprintf(ERR_BUF,"Unsupported datarep passed to MPI_File_set_view.\n");
        break;
    case 51:
        fprintf(ERR_BUF,"Unsupported operation, such as seeking on a file that supports only sequential access.\n");
        break;
    case 52:
        fprintf(ERR_BUF,"Invalid window.\n");
        break;
    case 53:
        fprintf(ERR_BUF,"Last error code.\n");
        break;
    case -2:
        fprintf(ERR_BUF,"Out of resources\n");
        break;
    }
}
#else  /* not using MPI */
/* read h5 file (single process) */
int read_hdf5(char filename[], int ndims,
              double etaS[NI+2][NJ+2][NK+2],
              double etaL[NI+2][NJ+2][NK+2],
              double c   [NI+2][NJ+2][NK+2]){
    /* HDF5 library variables */
    hid_t file_id,dataset_id_S,dataset_id_L,dataset_id_c;
    hid_t filespace_id,memspace_id;
    hsize_t h5dims[3],offset[3], stride[3],count[3],block[3];
    herr_t status;
    
    /* open the input file with default I/O access properties */
    file_id = H5Fopen(filename, H5F_ACC_RDONLY,H5P_DEFAULT);
    /* Open an existing dataset. */
    dataset_id_S = H5Dopen2(file_id, "/etaS",H5P_DEFAULT);
    /* create a memory dataspace and set the hyperslab */
    h5dims[0] = NI+2; h5dims[1] = NJ+2; h5dims[2] = NK+2;
    stride[0] = 1   ; stride[1] = 1   ; stride[2] = 1   ;
    count [0] = NI  ; count [1] = NJ  ; count [2] = NK  ;
    block [0] = 1   ; block [1] = 1   ; block [2] = 1   ;
    memspace_id = H5Screate_simple(ndims,h5dims,NULL);
    offset[0] = 1; offset[1] = 1; offset[2] = 1;
    status = H5Sselect_hyperslab(memspace_id,H5S_SELECT_SET,offset,stride,count,block);
    /* get the file dataspace and set the hyperslab */
    offset[0] = 0; offset[1] = 0; offset[2] = 0;
    filespace_id = H5Dget_space(dataset_id_S);
    status = H5Sselect_hyperslab(filespace_id,H5S_SELECT_SET,offset,stride,count,block);

    /* read a subset to etaS */
    H5Dread(dataset_id_S,H5T_IEEE_F64LE,memspace_id,filespace_id,H5P_DEFAULT,etaS);
    status = H5Sclose(filespace_id);
    status = H5Dclose(dataset_id_S);

    /* read a subset to etaL */
    dataset_id_L = H5Dopen2(file_id, "/etaL",H5P_DEFAULT);
    filespace_id = H5Dget_space(dataset_id_L);
    status = H5Sselect_hyperslab(filespace_id,H5S_SELECT_SET,offset,stride,count,block);
    H5Dread(dataset_id_S,H5T_IEEE_F64LE,memspace_id,filespace_id,H5P_DEFAULT,etaL);
    status = H5Sclose(filespace_id);
    status = H5Dclose(dataset_id_L);
    
    /* read a subset to c */
    dataset_id_c = H5Dopen2(file_id,    "/c",H5P_DEFAULT);
    filespace_id = H5Dget_space(dataset_id_c);
    status = H5Sselect_hyperslab(filespace_id,H5S_SELECT_SET,offset,stride,count,block);
    H5Dread(dataset_id_c,H5T_IEEE_F64LE,memspace_id,filespace_id,H5P_DEFAULT,c);
    status = H5Sclose(filespace_id);
    status = H5Dclose(dataset_id_c);

    /* close stuff */
    status = H5Sclose(memspace_id);
    status = H5Fclose(file_id);
    
    return 0;
}

/* write h5 file (single process) */
int write_hdf5(char filename[],int ndims,
               double etaS[NI+2][NJ+2][NK+2],
               double etaL[NI+2][NJ+2][NK+2],
               double c   [NI+2][NJ+2][NK+2]){
        /* HDF5 library variables */
    hid_t file_id,dataset_id_S,dataset_id_L,dataset_id_c;
    hid_t filespace_id,memspace_id;
    hsize_t h5dims[3],offset[3], stride[3],count[3],block[3];
    herr_t status;
    
    /* creat the output file with default I/O access properties */
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
    /* create dataspaces */
    h5dims[0] = NI+2; h5dims[1] = NJ+2; h5dims[2] = NK+2;
    memspace_id = H5Screate_simple(ndims,h5dims,NULL);
    h5dims[0] = NI; h5dims[1] = NJ; h5dims[2] = NK;
    filespace_id = H5Screate_simple(ndims,h5dims,NULL);
    /* create dataset */
    dataset_id_S = H5Dcreate2(file_id, "etaS", H5T_IEEE_F64LE, filespace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dataset_id_L = H5Dcreate2(file_id, "etaL", H5T_IEEE_F64LE, filespace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    dataset_id_c = H5Dcreate2(file_id,    "c", H5T_IEEE_F64LE, filespace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(filespace_id);
    /* set hyperslab */
    stride[0] = 1   ; stride[1] = 1   ; stride[2] = 1   ;
    count [0] = NI  ; count [1] = NJ  ; count [2] = NK  ;
    block [0] = 1   ; block [1] = 1   ; block [2] = 1   ;
    offset[0] = 1   ; offset[1] = 1   ; offset[2] = 1   ;
    status = H5Sselect_hyperslab(memspace_id,H5S_SELECT_SET,offset,stride,count,block);
    offset[0] = 0; offset[1] = 0; offset[2] = 0;
    filespace_id = H5Dget_space(dataset_id_S);
    status = H5Sselect_hyperslab(filespace_id,H5S_SELECT_SET,offset,stride,count,block);

    /* write etaS to file*/
    H5Dwrite(dataset_id_S,H5T_IEEE_F64LE,memspace_id,filespace_id,H5P_DEFAULT,etaS);
    status = H5Sclose(filespace_id);
    status = H5Dclose(dataset_id_S);

    /* write etaL to file */
    filespace_id = H5Dget_space(dataset_id_L);
    status = H5Sselect_hyperslab(filespace_id,H5S_SELECT_SET,offset,stride,count,block);
    H5Dwrite(dataset_id_L,H5T_IEEE_F64LE,memspace_id,filespace_id,H5P_DEFAULT,etaL);
    status = H5Sclose(filespace_id);
    status = H5Dclose(dataset_id_L);
    

    /* write c to file */
    filespace_id = H5Dget_space(dataset_id_c);
    status = H5Sselect_hyperslab(filespace_id,H5S_SELECT_SET,offset,stride,count,block);
    H5Dwrite(dataset_id_c,H5T_IEEE_F64LE,memspace_id,filespace_id,H5P_DEFAULT,c);
    status = H5Sclose(filespace_id);
    status = H5Dclose(dataset_id_c);

    /* close other stuff */
    status = H5Sclose(memspace_id);
    status = H5Fclose(file_id);

    return 0;
}

/* periodic boundary condition: not use MPI */
int apply_periodic_boundary_condition(double etaS[NI+2][NJ+2][NK+2],
                                      double etaL[NI+2][NJ+2][NK+2],
                                      double c   [NI+2][NJ+2][NK+2]){

    int i,j,k;
    for (j=1;j<=NJ;j++){
        for (k=1;k<=NK;k++){
            etaS[   0][j][k] = etaS[NI][j][k];
            etaL[   0][j][k] = etaL[NI][j][k];
            c   [   0][j][k] = c   [NI][j][k];
            etaS[NI+1][j][k] = etaS[ 1][j][k];
            etaL[NI+1][j][k] = etaL[ 1][j][k];
            c   [NI+1][j][k] = c   [ 1][j][k];
        }
    }
    for (i=1;i<=NI;i++){
        for (k=1;k<=NK;k++){
            etaS[i][   0][k] = etaS[i][NJ][k];
            etaL[i][   0][k] = etaL[i][NJ][k];
            c   [i][   0][k] = c   [i][NJ][k];
            etaS[i][NJ+1][k] = etaS[i][ 1][k];
            etaL[i][NJ+1][k] = etaL[i][ 1][k];
            c   [i][NJ+1][k] = c   [i][ 1][k];
        }
    }
    for (i=1;i<=NI;i++){
        for (j=1;j<=NJ;j++){
            etaS[i][j][   0] = etaS[i][j][NK];
            etaL[i][j][   0] = etaL[i][j][NK];
            c   [i][j][   0] = c   [i][j][NK];
            etaS[i][j][NK+1] = etaS[i][j][ 1];
            etaL[i][j][NK+1] = etaL[i][j][ 1];
            c   [i][j][NK+1] = c   [i][j][ 1];
        }
    }
    return 0;
}

/* non-flux boundary condition: not use MPI */
int apply_nonflux_boundary_condition(double etaS[NI+2][NJ+2][NK+2],
                                     double etaL[NI+2][NJ+2][NK+2],
                                     double c   [NI+2][NJ+2][NK+2]){

    int i,j,k;
    /* the middle between 0 and 1 (discussed with Yue Sun) */
    /* so the symmetry line is at the boundary of the cell */
    for (j=1;j<=NJ;j++){
        for (k=1;k<=NK;k++){
            etaS[   0][j][k] = etaS[ 1][j][k];
            etaL[   0][j][k] = etaL[ 1][j][k];
            c   [   0][j][k] = c   [ 1][j][k];
            etaS[NI+1][j][k] = etaS[NI][j][k];
            etaL[NI+1][j][k] = etaL[NI][j][k];
            c   [NI+1][j][k] = c   [NI][j][k];
        }
    }
    for (i=1;i<=NI;i++){
        for (k=1;k<=NK;k++){
            etaS[i][   0][k] = etaS[i][ 1][k];
            etaL[i][   0][k] = etaL[i][ 1][k];
            c   [i][   0][k] = c   [i][ 1][k];
            etaS[i][NJ+1][k] = etaS[i][NJ][k];
            etaL[i][NJ+1][k] = etaL[i][NJ][k];
            c   [i][NJ+1][k] = c   [i][NJ][k];
        }
    }
    for (i=1;i<=NI;i++){
        for (j=1;j<=NJ;j++){
            etaS[i][j][   0] = etaS[i][j][ 1];
            etaL[i][j][   0] = etaL[i][j][ 1];
            c   [i][j][   0] = c   [i][j][ 1];
            etaS[i][j][NK+1] = etaS[i][j][NK];
            etaL[i][j][NK+1] = etaL[i][j][NK];
            c   [i][j][NK+1] = c   [i][j][NK];
        }
    }
        return 0;
}
#endif    /* MPI */


int write_result_file(char *jobname, int save_iter,int ndims,
#if USE_MPI
                      MPI_Comm topocomm, MPI_Info info,
                      int coords[3],
#endif
                      double etaS[NI+2][NJ+2][NK+2],
                      double etaL[NI+2][NJ+2][NK+2],
                      double c[NI+2][NJ+2][NK+2]){
    char fname[50],str_iter[20];
    itoa(save_iter,str_iter);
    strcpy(fname,jobname); /* job name */
    strcat(fname,str_iter);
    strcat(fname,".h5");
    printf("%s\n",fname);
#if USE_MPI
    write_hdf5_parallel(fname,topocomm,info,ndims,coords,etaS,etaL,c);
#else  /* not using MPI */
    write_hdf5(fname,ndims,etaS,etaL,c);
#endif    /* MPI */
    return 0;
}
