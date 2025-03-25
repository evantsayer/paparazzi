/*
 * Copyright (C) 2025 Erik Stuttaford
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/everything_avoider_pf/everything_avoider_pf.h"
 * @brief Simplified potential fields based obstacle avoidance with NN integration.
 *
 * This module uses a neural network-based depth model (exported from PyTorch in ONNX format)
 * to compute a normalized direction value from the drone's front camera. The image is first
 * resized to 60x130 and then cropped (removing the left 20 columns) to yield a final input size
 * of 40x130. The NN outputs a value in [0,1] representing the desired direction
 * (0 = far left, 0.5 = straight ahead, 1 = far right). This value is used to adjust the drone's
 * heading and update its waypoints using a simplified state machine.
 */

 #ifndef EVERYTHING_AVOIDER_PF_H
 #define EVERYTHING_AVOIDER_PF_H
 
 #include "state.h"
 #include "generated/airframe.h"
 #include "firmwares/rotorcraft/navigation.h"  // For moveWaypointForward declaration
 
 /* Tunable Parameters (adjustable via settings XML) */
 extern float maxAngleDegrees;  // Maximum heading adjustment in degrees
 extern float moveDistance;     // Forward move distance in SAFE state
 extern float fallbackDistance; // Forward move distance in OUT_OF_BOUNDS state
 
 /* NN input dimensions (constants) */
 #define NN_INTERMEDIATE_WIDTH 60   // Image width after initial resize
 #define NN_FINAL_WIDTH        40   // Final image width after cropping left part
 #define NN_HEIGHT            130   // Image height (remains unchanged)
 
 /* Module State Enumeration */
 typedef enum {
   SAFE,
   OUT_OF_BOUNDS
 } navigation_state_t;
 
 /* External Functions for Module Control */
 void everything_avoider_pf_init(void);
 void everything_avoider_pf_periodic(void);
 
 /* External Function from navigation (if not already declared elsewhere) */
 extern void moveWaypointForward(uint8_t waypoint, float distanceMeters);
 
 #endif /* EVERYTHING_AVOIDER_PF_H */
 