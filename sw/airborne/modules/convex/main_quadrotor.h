#ifndef MAIN_QUADROTOR_H
#define MAIN_QUADROTOR_H

// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_quadrotor.h"
#include "subsystems/datalink/downlink.h"

// blasfeo
#include "blasfeo_d_aux_ext_dep.h"
#define PRINT(string,...) fprintf(stderr, "[project->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if DEBUG
  #define VERBOSE_PRINT PRINT
#else
  #define VERBOSE_PRINT(...)
#endif

#define NX     QUADROTOR_NX
#define NP     QUADROTOR_NP
#define NU     QUADROTOR_NU
#define NBX0   QUADROTOR_NBX0
#define NP_GLOBAL   QUADROTOR_NP_GLOBAL

int main();

#endif // MAIN_QUADROTOR_H
