#include "modules/orange_avoider_pf/orange_avoider_pf.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <math.h>

#define NAV_C  // needed to get functions like InsideObstacleZone()
#include "generated/flight_plan.h"

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider_pf->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
  #define VERBOSE_PRINT PRINT
#else
  #define VERBOSE_PRINT(...)
#endif

// ---------------- Static Function Declarations ----------------
static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static float compute_repulsive_adjustment(void);
static float fallback_increment_if_no_repulsion(float repulsive_adj);

// ---------------- Global Variables ----------------

// Potential fields parameters
float k_rep = 10.0f;  // Gain factor for repulsive force

// Obstacle detection parameters
float oa_color_count_frac = 0.18f; // Fraction of pixels that must be orange to trigger repulsion
int32_t color_count = 0;           // Current orange pixel count from the color filter
int16_t obstacle_center_x = 0;     // Horizontal position (x) of the detected obstacle

// Forward motion and safe confidence
int16_t obstacle_free_confidence = 0;
const int16_t max_safe_confidence = 10;
float maxDistance = 1.25f;         // Maximum waypoint displacement [m]

// State Machine
enum navigation_state_t {
  SAFE,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS
};
enum navigation_state_t navigation_state = SAFE;

// ---------------- ABI Callback ----------------
#ifndef ORANGE_AVOIDER_PF_VISUAL_DETECTION_ID
  #define ORANGE_AVOIDER_PF_VISUAL_DETECTION_ID ABI_BROADCAST
#endif

static abi_event color_detection_ev;
static void color_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               int16_t pixel_x, int16_t __attribute__((unused)) pixel_y,
                               int16_t __attribute__((unused)) pixel_width, int16_t __attribute__((unused)) pixel_height,
                               int32_t quality, int16_t __attribute__((unused)) extra)
{
  color_count = quality;
  obstacle_center_x = pixel_x; // Save horizontal position of the detected obstacle
}

// ---------------- Module Initialization ----------------
void orange_avoider_pf_init(void)
{
  srand(time(NULL));
  // Bind callback for obstacle detection
  AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_PF_VISUAL_DETECTION_ID, &color_detection_ev, color_detection_cb);

  navigation_state = SAFE;
  obstacle_free_confidence = 0;
}

// ---------------- Main Periodic Function ----------------
void orange_avoider_pf_periodic(void)
{
  if (!autopilot_in_flight()) {
    return;
  }

  // Compute threshold for detecting orange pixels
  int32_t total_pixels = front_camera.output_size.w * front_camera.output_size.h;
  int32_t color_threshold = oa_color_count_frac * total_pixels;

  VERBOSE_PRINT("Color_count: %d, Threshold: %d, ObstacleCenter_x: %d, State: %d\n",
                color_count, color_threshold, obstacle_center_x, navigation_state);

  // Update safe confidence
  if (color_count < color_threshold) {
    if (obstacle_free_confidence < max_safe_confidence) {
      obstacle_free_confidence++;
    }
  } else {
    obstacle_free_confidence -= 2;
    if (obstacle_free_confidence < 0) {
      obstacle_free_confidence = 0;
    }
  }

  // Calculate forward distance based on safe confidence
  float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);

  // Compute repulsive heading adjustment from the obstacle
  float repulsive_adj = compute_repulsive_adjustment();

  // ---------------- State Machine ----------------
  switch (navigation_state) {

    case SAFE: {
      // Apply repulsive heading (0 if no obstacle)
      increase_nav_heading(repulsive_adj);

      // Move the trajectory waypoint forward
      moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);

      // If WP_TRAJECTORY is out-of-bounds => go to OUT_OF_BOUNDS
      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;
        VERBOSE_PRINT("Switching state to OUT_OF_BOUNDS\n");
      }
      // If no safe confidence => OBSTACLE_FOUND
      else if (obstacle_free_confidence == 0) {
        navigation_state = OBSTACLE_FOUND;
        VERBOSE_PRINT("Switching state to OBSTACLE_FOUND\n");
      }
      else {
        // Update GOAL and RETREAT
        moveWaypointForward(WP_GOAL, moveDistance);
        moveWaypointForward(WP_RETREAT, -1.0f * moveDistance);
      }
      break;
    }

    case OBSTACLE_FOUND: {
      // Stop forward motion
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_RETREAT);
      waypoint_move_here_2d(WP_TRAJECTORY);

      // Instead of random increments, use repulsive to reorient
      increase_nav_heading(repulsive_adj);

      // Once we have a bit of safe confidence, switch to SEARCH
      if (obstacle_free_confidence >= 2) {
        navigation_state = SEARCH_FOR_SAFE_HEADING;
        VERBOSE_PRINT("Switching state to SEARCH_FOR_SAFE_HEADING\n");
      }
      break;
    }

    case SEARCH_FOR_SAFE_HEADING: {
      // Keep applying repulsive heading
      increase_nav_heading(repulsive_adj);

      // If safe confidence is built up, go back to SAFE
      if (obstacle_free_confidence >= 2) {
        navigation_state = SAFE;
        VERBOSE_PRINT("Switching state to SAFE\n");
      }
      break;
    }

    case OUT_OF_BOUNDS: {
      // If there's no obstacle, repulsive_adj = 0 => fallback
      float heading_correction = fallback_increment_if_no_repulsion(repulsive_adj);
      increase_nav_heading(heading_correction);

      // Move a small distance forward to re-enter
      moveWaypointForward(WP_TRAJECTORY, 1.5f);
      moveWaypointForward(WP_RETREAT, -1.0f);

      // If WP_TRAJECTORY is now inside => reset confidence => SEARCH
      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        obstacle_free_confidence = 0;
        navigation_state = SEARCH_FOR_SAFE_HEADING;
        VERBOSE_PRINT("Back in bounds => SEARCH_FOR_SAFE_HEADING\n");
      }
      break;
    }

    default:
      break;
  }
}

// ---------------- Helper Functions ----------------

// Compute the repulsive heading adjustment in degrees
static float compute_repulsive_adjustment(void)
{
  // If color_count < threshold => no repulsion => 0
  // Otherwise => k_rep * normalized offset
  // This logic is consistent with your original approach
  // but returns the heading increment in degrees
  int32_t total_pixels = front_camera.output_size.w * front_camera.output_size.h;
  int32_t color_threshold = oa_color_count_frac * total_pixels;
  if (color_count < color_threshold) {
    return 0.0f;
  }

  float repulsive_magnitude = (color_count - color_threshold) / (float)color_threshold;
  float image_center = front_camera.output_size.w / 2.0f;
  float offset = (image_center - obstacle_center_x) / image_center; // [-1,1]
  return k_rep * repulsive_magnitude * offset;
}

// Fallback increment if repulsive adjustment == 0
static float fallback_increment_if_no_repulsion(float repulsive_adj)
{
  // If there's no obstacle => repulsive_adj=0 => use a small heading increment to re-enter
  // Otherwise => just use the repulsive adjustment
  if (fabsf(repulsive_adj) < 1e-3f) {
    // fallback of 5 degrees
    return 10.0f;
  }
  return repulsive_adj;
}

// Increase heading by incrementDegrees (in deg)
uint8_t increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;
  VERBOSE_PRINT("Increased heading by %f deg => new heading: %f deg\n",
                incrementDegrees, DegOfRad(new_heading));
  return false;
}

// Move a waypoint forward by distanceMeters
uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

// Calculate new coordinates forward from current position & heading
uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading = stateGetNedToBodyEulers_f()->psi;
  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * distanceMeters);
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * distanceMeters);
  VERBOSE_PRINT("Forward %f => x:%f, y:%f\n",
                distanceMeters, POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y));
  return false;
}

// Update the waypoint position
uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  VERBOSE_PRINT("Moving WP %d => x:%f, y:%f\n",
                waypoint, POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y));
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}


