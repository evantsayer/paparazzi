#include "green_finder.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <math.h>

#define NAV_C
#include "generated/flight_plan.h"

// Tuning parameters for green detection (attraction)
float k_attr = 40.0f;               // Attractive gain factor
float maxDistance = 1.50f;           // Maximum waypoint movement distance
float gf_green_count_frac = 0.15f;   // Fraction of pixels that must be green to be considered safe

// Global variables for detection
volatile int32_t color_count = 0;
volatile int16_t green_center_x = 0;
int16_t safe_confidence = 0;
const int16_t max_safe_confidence = 10;

// State machine for navigation based on green detection
enum navigation_state_t {
  GREEN_SAFE,
  GREEN_LOST,
  SEARCH_FOR_GREEN_HEADING,
  OUT_OF_BOUNDS
};
enum navigation_state_t navigation_state = GREEN_SAFE;

// Planning update counter: Only update planning every 3 cycles (~1.33Hz if periodic is 4Hz)
static int planning_counter = 0;

// ABI callback for visual green detection
static abi_event color_detection_ev;
static void green_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               int16_t pixel_x, int16_t __attribute__((unused)) pixel_y,
                               int16_t __attribute__((unused)) pixel_width, int16_t __attribute__((unused)) pixel_height,
                               int32_t quality, int16_t __attribute__((unused)) extra)
{
  color_count = quality;
  green_center_x = pixel_x;
}

// Module initialization: bind the visual detection message and reset state
void green_finder_init(void)
{
  srand((unsigned)time(NULL));
  AbiBindMsgVISUAL_DETECTION(ABI_BROADCAST, &color_detection_ev, green_detection_cb);
  navigation_state = GREEN_SAFE;
  safe_confidence = 0;
  planning_counter = 0;
}

// Main periodic function: update safe confidence, compute attractive adjustment,
// and update waypoints/heading based on the current state.
void green_finder_periodic(void)
{
  if (!autopilot_in_flight()) return;

  // Increase planning counter and only update every 3 cycles.
  planning_counter++;
  if (planning_counter < 3) {
    return;
  }
  planning_counter = 0;

  const int width = front_camera.output_size.w;
  const int height = front_camera.output_size.h;
  const int32_t total_pixels = width * height;
  const int32_t color_threshold = (int32_t)(gf_green_count_frac * total_pixels);

  // Update safe confidence based on green detection (more green = safer)
  if (color_count >= color_threshold) {
    if (safe_confidence < max_safe_confidence) safe_confidence++;
  } else {
    safe_confidence = (safe_confidence > 2) ? safe_confidence - 2 : 0;
  }

  const float moveDistance = fminf(maxDistance, 0.2f * safe_confidence);
  const float attractive_adj = compute_attractive_adjustment(color_threshold);

  // State Machine:
  switch (navigation_state) {
    case GREEN_SAFE:
      increase_nav_heading(attractive_adj);
      moveWaypointForward(WP_TRAJECTORY, 1.0f * moveDistance);
      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;
      } else if (safe_confidence == 0) {
        navigation_state = GREEN_LOST;
      } else {
        moveWaypointForward(WP_GOAL, moveDistance);
        moveWaypointForward(WP_RETREAT, -moveDistance);
      }
      break;

    case GREEN_LOST:
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_RETREAT);
      waypoint_move_here_2d(WP_TRAJECTORY);
      // If the attractive adjustment is negligible or safe_confidence is low, do a fixed 15° rotation.
      if (fabsf(attractive_adj) < 1.0f || safe_confidence < 2) {
        increase_nav_heading(15.0f);
      } else {
        increase_nav_heading(attractive_adj);
      }
      if (safe_confidence >= 2) navigation_state = SEARCH_FOR_GREEN_HEADING;
      break;
    

    case SEARCH_FOR_GREEN_HEADING:
      increase_nav_heading(attractive_adj);
      if (safe_confidence >= 2) navigation_state = GREEN_SAFE;
      break;

    case OUT_OF_BOUNDS:
      // If the attractive adjustment is too small or safe confidence is low,
      // perform a fixed 15° search rotation.
      if (fabsf(attractive_adj) < 1.0f || safe_confidence < 2) {
        increase_nav_heading(15.0f);
      } else {
        increase_nav_heading(attractive_adj);
      }
      moveWaypointForward(WP_TRAJECTORY, 1.5f);
      moveWaypointForward(WP_RETREAT, -1.0f);
      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        safe_confidence = 0;
        navigation_state = SEARCH_FOR_GREEN_HEADING;
      }
      break;
  }
}

// Computes an attractive adjustment to steer the drone toward the green area.
// The adjustment scales with how much the green pixel count exceeds the threshold,
// and the horizontal offset of the green region from the image center.
float compute_attractive_adjustment(int32_t color_threshold)
{
  if (color_count < color_threshold) return 0.0f;
  float attractive_magnitude = ((color_count - color_threshold) / (float) color_threshold);
  float image_center = front_camera.output_size.w * 0.5f;
  float offset = (green_center_x - image_center) / image_center;
  return k_attr * attractive_magnitude * offset;
}

// Provides a fallback increment when the computed attractive adjustment is near zero.
float fallback_increment_if_no_attraction(float attractive_adj)
{
  return (fabsf(attractive_adj) < 1e-3f) ? 10.0f : attractive_adj;
}

// Adjusts the navigation heading by a given increment (in degrees),
// but clamps the change to avoid excessive spinning.
void increase_nav_heading(float incrementDegrees)
{
  float current_heading = stateGetNedToBodyEulers_f()->psi;  // current heading in radians
  // Convert desired increment to radians.
  float desired_change = RadOfDeg(incrementDegrees);
  // Clamp the heading change to a maximum value (e.g., 5 degrees per update).
  float max_heading_change = RadOfDeg(5.0f);
  if (desired_change > max_heading_change)
    desired_change = max_heading_change;
  else if (desired_change < -max_heading_change)
    desired_change = -max_heading_change;

  float new_heading = current_heading + desired_change;
  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;
}

// Helper: compute a new waypoint forward based on the current position and heading.
void calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  const struct FloatEulers* eulers = stateGetNedToBodyEulers_f();
  const struct EnuCoor_i* pos = stateGetPositionEnu_i();
  float sin_h = sinf(eulers->psi);
  float cos_h = cosf(eulers->psi);
  new_coor->x = pos->x + POS_BFP_OF_REAL(sin_h * distanceMeters);
  new_coor->y = pos->y + POS_BFP_OF_REAL(cos_h * distanceMeters);
}

// Updates a waypoint to a new XY coordinate.
void moveWaypoint(uint8_t waypoint, const struct EnuCoor_i *new_coor)
{
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
}

// Moves a waypoint forward by a given distance.
void moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
}
