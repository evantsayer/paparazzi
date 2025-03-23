/*
 * Copyright (C) 2025 Gabriel Gervas
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider_polar/orange_avoider_polar.h"
 * @author Gabriel Gervas
 * 
 */

 #ifndef ORANGE_AVOIDER_PF_H
 #define ORANGE_AVOIDER_PF_H
 
 // Include necessary libraries
 #include "state.h"
 #include "generated/airframe.h"
 
 // Settings (these are linked to XML settings for tuning)
 extern float oa_color_count_frac;  // Threshold fraction for orange pixel detection
 extern float maxDistance;          // Max waypoint movement distance
 extern float k_rep;                // Gain factor for repulsive force computation
 
 // Functions to be used externally
 extern void orange_avoider_polar_init(void);     // Initialization function
 extern void orange_avoider_polar_periodic(void); // Periodic function for control logic

 // Function prototypes
 void orange_avoider_polar_init(void);
 void orange_avoider_polar_periodic(void);
 float compute_repulsive_adjustment(int32_t color_threshold);
 float fallback_increment_if_no_repulsion(float repulsive_adj);
 void increase_nav_heading(float incrementDegrees);
 void moveWaypointForward(uint8_t waypoint, float distanceMeters);

 
 #endif /* ORANGE_AVOIDER_PF_H */
 