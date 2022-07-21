
#ifndef _POWSPEC_H_
#define _POWSPEC_H_


#include "read_cata.h"
#include "multipole.h"

void free_pk_array(double *pk_array);
int compute_pk(CATA *cata, int *nkbin, double *pk_array, int argc, char *argv[]);

#endif