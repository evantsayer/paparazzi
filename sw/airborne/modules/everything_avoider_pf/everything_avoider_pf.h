/*
 * Copyright (C) 2025 Erik Stuttaford
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/everything_avoider_pf/everything_avoider_pf.h"
 * @brief Simplified potential fields based obstacle avoidance with NN integration.
 */

 #ifndef EVERYTHING_AVOIDER_PF_H
 #define EVERYTHING_AVOIDER_PF_H
 
 #include "state.h"
 #include "generated/airframe.h"
 #include "firmwares/rotorcraft/navigation.h"  // For moveWaypointForward
 
 /* NN input dimensions (constants) */
 #define NN_INTERMEDIATE_WIDTH 60
 #define NN_FINAL_WIDTH        40
 #define NN_HEIGHT             130
 
 /* Tunable Parameters (adjustable via settings XML) */
 extern float maxAngleDegrees;
 extern float moveDistance;
 extern float fallbackDistance;
 
 /* Module State Enumeration */
 typedef enum {
   SAFE,
   OUT_OF_BOUNDS
 } navigation_state_t;
 
 /* Function declarations */
 #ifdef __cplusplus
 extern "C" {
 #endif
 
 void everything_avoider_pf_init(void);
 void everything_avoider_pf_periodic(void);
 void moveWaypointForward(uint8_t waypoint, float distanceMeters);
 void get_camera_image_normalized(float* final_buffer);
 
 void increase_nav_heading(float incrementDegrees);
 void calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
 void moveWaypoint(uint8_t waypoint, const struct EnuCoor_i *new_coor);
 
 #ifdef __cplusplus
 }
 #endif
 
 #endif /* EVERYTHING_AVOIDER_PF_H */
 