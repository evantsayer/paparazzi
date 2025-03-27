/*
 * Copyright (c) The acados authors.
 *
 * The 2-Clause BSD License
 */

// Standard includes
#include "main_quadrotor.h"
#include "subsystems/datalink/downlink.h"
#include <stdio.h>
#include <stdlib.h>

// Enable debug logging
#define DEBUG 1  

// Macros for logging
#define PRINT(string,...) fprintf(stderr, "[project->%s()] " string "\n", __FUNCTION__, ##__VA_ARGS__)

#if DEBUG
  #define VERBOSE_PRINT PRINT
#else
  #define VERBOSE_PRINT(...)
#endif

// Define quadrotor dimensions
#define NX     QUADROTOR_NX
#define NP     QUADROTOR_NP
#define NU     QUADROTOR_NU
#define NBX0   QUADROTOR_NBX0
#define NP_GLOBAL   QUADROTOR_NP_GLOBAL

int main()
{
    VERBOSE_PRINT("Starting quadrotor program...");

    quadrotor_solver_capsule *acados_ocp_capsule = quadrotor_acados_create_capsule();
    int N = QUADROTOR_N;
    double* new_time_steps = NULL;

    int status = quadrotor_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);
    if (status)
    {
        printf("quadrotor_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    ocp_nlp_config *nlp_config = quadrotor_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = quadrotor_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = quadrotor_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = quadrotor_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = quadrotor_acados_get_nlp_solver(acados_ocp_capsule);

    // Initial conditions
    double lbx0[NBX0] = {1, 1, 1, 0, 0, 0};
    double ubx0[NBX0] = {1, 1, 1, 0, 0, 0};

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);

    // State initialization
    double x_init[NX] = {0.0};
    double u0[NU] = {0.0};

    int NTIMINGS = 1;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;
    double xtraj[NX * (N+1)];
    double utraj[NU * N];

    VERBOSE_PRINT("Starting OCP solver loop...");
    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
        }
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x_init);
        status = quadrotor_acados_solve(acados_ocp_capsule);
        ocp_nlp_get(nlp_solver, "time_tot", &elapsed_time);
        min_time = (elapsed_time < min_time) ? elapsed_time : min_time;
    }

    // Print solution
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &xtraj[ii*NX]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &utraj[ii*NU]);

    printf("\n--- xtraj ---\n");
    d_print_exp_tran_mat(NX, N+1, xtraj, NX);
    printf("\n--- utraj ---\n");
    d_print_exp_tran_mat(NU, N, utraj, NU);
    printf("\nsolved ocp %d times, solution printed above\n\n", NTIMINGS);

    if (status == ACADOS_SUCCESS)
    {
        printf("quadrotor_acados_solve(): SUCCESS!\n");
    }
    else
    {
        printf("quadrotor_acados_solve() failed with status %d.\n", status);
    }

    // Get solver stats
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_solver, "sqp_iter", &sqp_iter);

    quadrotor_acados_print_stats(acados_ocp_capsule);

    VERBOSE_PRINT("Solver info:");
    VERBOSE_PRINT("SQP iterations %d, minimum time for %d solve: %.3f ms, KKT: %e",
             sqp_iter, NTIMINGS, min_time * 1000, kkt_norm_inf);

    // Cleanup
    status = quadrotor_acados_free(acados_ocp_capsule);
    if (status) {
        printf("quadrotor_acados_free() returned status %d. \n", status);
    }

    status = quadrotor_acados_free_capsule(acados_ocp_capsule);
    if (status) {
        printf("quadrotor_acados_free_capsule() returned status %d. \n", status);
    }

    return status;
}
