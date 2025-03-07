#include "modules/orange_avoider/orange_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <math.h>

#define NAV_C // needed to get the nav functions like Inside...
#include "generated/flight_plan.h"

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

// Global variables for potential fields-based avoidance
// Fraction of pixels that must be orange to trigger repulsion
float oa_color_count_frac = 0.18f;
int32_t color_count = 0;                // orange color count from the color filter (quality field)
int16_t obstacle_center_x = 0;          // horizontal center of detected obstacle (from color filter)

// You can still keep a notion of safe confidence if you wish to gradually ramp forward motion.
int16_t obstacle_free_confidence = 0;
float maxDistance = 2.25;               // maximum forward waypoint displacement [m]

//----------------------------------------------------------------------
// Updated ABI callback: now we store the pixel x coordinate to know where the obstacle is.
//----------------------------------------------------------------------
#ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
#define ORANGE_AVOIDER_VISUAL_DETECTION_ID ABI_BROADCAST
#endif
static abi_event color_detection_ev;
static void color_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               int16_t pixel_x, int16_t __attribute__((unused)) pixel_y,
                               int16_t __attribute__((unused)) pixel_width, int16_t __attribute__((unused)) pixel_height,
                               int32_t quality, int16_t __attribute__((unused)) extra)
{
  color_count = quality;
  obstacle_center_x = pixel_x;  // save the horizontal location of the detected obstacle
}

//----------------------------------------------------------------------
// Initialisation function remains similar
//----------------------------------------------------------------------
void orange_avoider_init(void)
{
  // Initialise random seed if needed (not used in this potential fields version)
  srand(time(NULL));

  // Bind our color filter callback to receive outputs.
  AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, &color_detection_ev, color_detection_cb);
}

//----------------------------------------------------------------------
// Updated periodic function using potential fields method.
//----------------------------------------------------------------------
// This version removes the state-machine and instead computes a repulsive heading adjustment.
// When the detected orange pixel count exceeds a threshold, a repulsive force is computed that
// causes the drone to turn away from the obstacle. When the measurement is below threshold, the drone
// continues forward.
void orange_avoider_periodic(void)
{
  if (!autopilot_in_flight()){
    return;
  }

  // Compute the threshold based on the fraction and the total number of pixels from the front camera.
  int32_t total_pixels = front_camera.output_size.w * front_camera.output_size.h;
  int32_t color_threshold = oa_color_count_frac * total_pixels;

  VERBOSE_PRINT("Color_count: %d, Threshold: %d, ObstacleCenter_x: %d\n", color_count, color_threshold, obstacle_center_x);

  // Define a gain factor (in degrees) for the repulsive force
  float k_rep = 10.0f;
  float heading_adjustment = 0.0f;

  // If the number of orange pixels exceeds the threshold, compute a repulsive force.
  if (color_count >= color_threshold) {
    // Compute a normalized repulsive magnitude based on the excess of orange pixels.
    float repulsive_magnitude = (color_count - color_threshold) / (float)color_threshold;

    // Determine the horizontal offset from the center of the image.
    // If obstacle_center_x is less than half the image width, the obstacle is on the left.
    float image_center = front_camera.output_size.w / 2.0f;
    float offset = (image_center - obstacle_center_x) / image_center; // normalized to [-1, 1]

    // The repulsive heading adjustment (in degrees) is proportional to both the excess and the offset.
    heading_adjustment = k_rep * repulsive_magnitude * offset;
    VERBOSE_PRINT("Repulsive heading adjustment: %f degrees\n", heading_adjustment);
  } else {
    // No significant obstacle: optionally, you might let the heading settle back to 0 adjustment.
    heading_adjustment = 0.0f;
  }

  // Update the navigation heading by applying the computed adjustment.
  increase_nav_heading(heading_adjustment);

  // Determine forward motion:
  // If the obstacle is below threshold, proceed forward.
  // Otherwise, hold position (or you might add a slight retreat).
  if (color_count < color_threshold) {
    // For example, move forward by a distance proportional to a safe confidence measure.
    // (Here, obstacle_free_confidence could be updated based on multiple consecutive safe readings.)
    float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);
    moveWaypointForward(WP_GOAL, moveDistance);
  } else {
    // If an obstacle is present, do not move forward.
    waypoint_move_here_2d(WP_GOAL);
  }
}

//----------------------------------------------------------------------
// The remaining functions (increase_nav_heading, moveWaypointForward, calculateForwards, moveWaypoint)
// remain unchanged. They are used to update the heading and waypoint positions based on computed values.
//----------------------------------------------------------------------

uint8_t increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
  // Normalize heading to [-pi, pi]
  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;
  VERBOSE_PRINT("Updated heading to %f degrees\n", DegOfRad(new_heading));
  return false;
}

uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading = stateGetNedToBodyEulers_f()->psi;
  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * (distanceMeters));
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * (distanceMeters));
  VERBOSE_PRINT("Calculated forward position: %f m. New coordinates: x: %f, y: %f\n",
                distanceMeters, POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y));
  return false;
}

uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  VERBOSE_PRINT("Moving waypoint %d to x: %f, y: %f\n", waypoint,
                POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y));
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}
