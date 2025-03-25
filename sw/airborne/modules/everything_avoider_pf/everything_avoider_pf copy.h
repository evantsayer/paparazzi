/*
 * Copyright (C) 2025 Erik Stuttaford
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/everything_avoider_pf/everything_avoider_pf.h"
 * @author Erik Stuttaford
 *
 * Example implementation of potential fields for obstacle avoidance.
 * This module now integrates a neural network depth model (exported from PyTorch as "depth_model.onnx")
 * to compute a depth map from the drone's front camera. The depth map is processed to detect obstacles
 * by counting high-valued (red) pixels. The avoidance logic is based on potential fields.
 */

 #ifndef EVERYTHING_AVOIDER_PF_H
 #define EVERYTHING_AVOIDER_PF_H
  
 // Include necessary libraries
 #include "state.h"
 #include "generated/airframe.h"
  
 // Settings (these are linked to XML settings for tuning)
 // Note: oa_color_count_frac now represents the threshold fraction for red pixel detection
 // from the NN-generated depth map (instead of solely detecting orange pixels).
 extern float oa_color_count_frac;  // Threshold fraction for red pixel detection
 extern float maxDistance;          // Max waypoint movement distance
 extern float k_rep;                // Gain factor for repulsive force computation
  
 // Functions to be used externally
 extern void everything_avoider_pf_init(void);     // Initialization function
 extern void everything_avoider_pf_periodic(void);  // Periodic function for control logic
 
 // Function prototypes
 void everything_avoider_pf_init(void);
 void everything_avoider_pf_periodic(void);
 float compute_repulsive_adjustment(int32_t red_threshold);
 float fallback_increment_if_no_repulsion(float repulsive_adj);
 void increase_nav_heading(float incrementDegrees);
 void moveWaypointForward(uint8_t waypoint, float distanceMeters);
  
 #endif /* EVERYTHING_AVOIDER_PF_H */
 