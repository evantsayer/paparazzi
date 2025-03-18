#ifndef GREEN_FINDER_H
#define GREEN_FINDER_H

/*
 * Copyright (C) 2025 Erik Stuttaford
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/green_finder/green_finder.h"
 * Updated for green grass detection and navigation.
 */

#include "state.h"
#include "generated/airframe.h"

// Settings (linked to XML settings for tuning)
extern float gf_green_count_frac;  // Threshold fraction for green pixel detection
extern float maxDistance;          // Maximum waypoint movement distance
extern float k_attr;               // Gain factor for attractive force computation


// Functions available externally
extern void green_finder_init(void);
extern void green_finder_periodic(void);

// Function prototypes
void green_finder_init(void);
void green_finder_periodic(void);
float compute_attractive_adjustment(int32_t color_threshold);
float fallback_increment_if_no_attraction(float attractive_adj);
void increase_nav_heading(float incrementDegrees);
void moveWaypointForward(uint8_t waypoint, float distanceMeters);

#endif /* GREEN_FINDER_H */

 