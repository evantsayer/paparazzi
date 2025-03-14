#include "modules/orange_avoider_pf/orange_avoider_pf.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <math.h>

#define NAV_C  
#include "generated/flight_plan.h"

// ---------------- Global Variables (Declared in Header) ----------------
float k_rep = 40.0f;
float maxDistance = 1.50f;
float oa_color_count_frac = 0.15f; //Percent of pixels that must be orange for an obstacle to be flagged

volatile int32_t color_count = 0;
volatile int16_t obstacle_center_x = 0;
int16_t obstacle_free_confidence = 0;
const int16_t max_safe_confidence = 10;

enum navigation_state_t {
  SAFE,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS
};
enum navigation_state_t navigation_state = SAFE;

// ---------------- ABI Callback ----------------
static abi_event color_detection_ev;

static void color_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               int16_t pixel_x, int16_t __attribute__((unused)) pixel_y,
                               int16_t __attribute__((unused)) pixel_width, int16_t __attribute__((unused)) pixel_height,
                               int32_t quality, int16_t __attribute__((unused)) extra)
{
  color_count = quality;
  obstacle_center_x = pixel_x;
}

// ---------------- Module Initialization ----------------
void orange_avoider_pf_init(void)
{
  srand((unsigned)time(NULL));
  AbiBindMsgVISUAL_DETECTION(ABI_BROADCAST, &color_detection_ev, color_detection_cb);
  navigation_state = SAFE;
  obstacle_free_confidence = 0;
}

// ---------------- Main Periodic Function ----------------
void orange_avoider_pf_periodic(void)
{
  if (!autopilot_in_flight()) return;

  const int width = front_camera.output_size.w;
  const int height = front_camera.output_size.h;
  const int32_t total_pixels = width * height;
  const int32_t color_threshold = (int32_t)(oa_color_count_frac * total_pixels);
  const float image_center = width * 0.5f;

  // Safe confidence update
  if (color_count < color_threshold) {
    if (obstacle_free_confidence < max_safe_confidence) obstacle_free_confidence++;
  } else {
    obstacle_free_confidence = (obstacle_free_confidence > 2) ? obstacle_free_confidence - 2 : 0;
  }

  const float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);
  const float repulsive_adj = compute_repulsive_adjustment(color_threshold);

  // ---------------- State Machine ----------------
  switch (navigation_state) {
    case SAFE:
      increase_nav_heading(repulsive_adj);
      moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);

      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;
      } else if (obstacle_free_confidence == 0) {
        navigation_state = OBSTACLE_FOUND;
      } else {
        moveWaypointForward(WP_GOAL, moveDistance);
        moveWaypointForward(WP_RETREAT, -moveDistance);
      }
      break;

    case OBSTACLE_FOUND:
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_RETREAT);
      waypoint_move_here_2d(WP_TRAJECTORY);
      increase_nav_heading(repulsive_adj);
      if (obstacle_free_confidence >= 2) navigation_state = SEARCH_FOR_SAFE_HEADING;
      break;

    case SEARCH_FOR_SAFE_HEADING:
      increase_nav_heading(repulsive_adj);
      if (obstacle_free_confidence >= 2) navigation_state = SAFE;
      break;

    case OUT_OF_BOUNDS:
      increase_nav_heading(fallback_increment_if_no_repulsion(repulsive_adj));
      moveWaypointForward(WP_TRAJECTORY, 1.5f);
      moveWaypointForward(WP_RETREAT, -1.0f);

      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        obstacle_free_confidence = 0;
        navigation_state = SEARCH_FOR_SAFE_HEADING;
      }
      break;
  }
}

// ---------------- Helper Functions ----------------
float compute_repulsive_adjustment(int32_t color_threshold)
{
  if (color_count < color_threshold) return 0.0f;

  float repulsive_magnitude = ((color_count - color_threshold) * (1.0f / color_threshold));
  float offset = (front_camera.output_size.w * 0.5f - obstacle_center_x) * (1.0f / (front_camera.output_size.w * 0.5f));
  return k_rep * repulsive_magnitude * offset;
}

float fallback_increment_if_no_repulsion(float repulsive_adj)
{
  return (fabsf(repulsive_adj) < 1e-3f) ? 10.0f : repulsive_adj;
}

void increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;
}

void calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  const struct FloatEulers* eulers = stateGetNedToBodyEulers_f();
  const struct EnuCoor_i* pos = stateGetPositionEnu_i();
  float sin_h = sinf(eulers->psi);
  float cos_h = cosf(eulers->psi);

  new_coor->x = pos->x + POS_BFP_OF_REAL(sin_h * distanceMeters);
  new_coor->y = pos->y + POS_BFP_OF_REAL(cos_h * distanceMeters);
}

void moveWaypoint(uint8_t waypoint, const struct EnuCoor_i *new_coor)
{
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
}

void moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
}
