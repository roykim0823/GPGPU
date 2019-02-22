#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main( int argc, char *argv[] )
{
    int provided, claimed;
          
    /*** Select one of the following
        MPI_Init_thread( 0, 0, MPI_THREAD_SINGLE, &provided );
        MPI_Init_thread( 0, 0, MPI_THREAD_FUNNELED, &provided );
        MPI_Init_thread( 0, 0, MPI_THREAD_SERIALIZED, &provided );
        MPI_Init_thread( 0, 0, MPI_THREAD_MULTIPLE, &provided );
    ***/ 
    int i=atoi(argv[1]);
    switch(i) {
        case 0: MPI_Init_thread( 0, 0, MPI_THREAD_SINGLE, &provided );      break;
        case 1: MPI_Init_thread( 0, 0, MPI_THREAD_FUNNELED, &provided );    break;
        case 2: MPI_Init_thread( 0, 0, MPI_THREAD_SERIALIZED, &provided );  break;
        case 3: MPI_Init_thread( 0, 0, MPI_THREAD_MULTIPLE, &provided );    break;
    }
    MPI_Query_thread( &claimed );
    printf( "Query thread level= %d  Init_thread level= %d\n", claimed, provided );
    MPI_Finalize();
}
