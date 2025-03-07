/*
 * Copyright (C) 2025 Erik Stuttaford
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider_pf/orange_avoider_pf.h"
 * @author Erik Stuttaford
 * Example implementation of potential fields for obstacle avoidance
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
 extern void orange_avoider_pf_init(void);     // Initialization function
 extern void orange_avoider_pf_periodic(void); // Periodic function for control logic
 
 #endif /* ORANGE_AVOIDER_PF_H */
 