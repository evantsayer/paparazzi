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
float k_attr = 13.5f;               // Attractive gain factor
float maxDistance = 1.8f;           // Maximum waypoint movement distance
float gf_green_count_frac = 0.28f;   // Fraction of pixels that must be green to be considered safe
float gf_unsafe_green_count_frac = 0.22f; // Fraction of pixels below which it is very unsafe

// Global variables for detection
volatile int32_t color_count = 0;
volatile int16_t green_center_x = 0;
volatile int16_t green_center_y = 0;
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
static int planning_counter = 1;
static int waypoint_update_counter = 0; // Counter to track updates since the last waypoint change

// ABI callback for visual green detection
static abi_event color_detection_ev;
static void green_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               int16_t pixel_x, int16_t pixel_y,
                               int16_t __attribute__((unused)) pixel_width, int16_t __attribute__((unused)) pixel_height,
                               int32_t quality, int16_t __attribute__((unused)) extra)
{
    color_count = quality;
    green_center_x = pixel_x;
    green_center_y = pixel_y; // Update vertical center
}

// Module initialization: bind the visual detection message and reset state
void green_finder_init(void)
{
  srand((unsigned)time(NULL));
  AbiBindMsgVISUAL_DETECTION(ABI_BROADCAST, &color_detection_ev, green_detection_cb);
  navigation_state = GREEN_LOST;
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
    const int32_t unsafe_color_threshold = (int32_t)(gf_unsafe_green_count_frac * total_pixels);

    // Debug: Log green detection stats
    printf("[green_finder -> green_finder_periodic()] Color count: %d, Threshold: %d, Unsafe threshold: %d, Safe confidence: %d\n",
           color_count, color_threshold, unsafe_color_threshold, safe_confidence);



    // Update safe confidence based on green detection
    if (color_count >= color_threshold) {
        if (safe_confidence < max_safe_confidence) safe_confidence++;
    } else {
        safe_confidence = (safe_confidence > 3) ? safe_confidence - 3 : 0;
    }

    // Debug: Log updated safe confidence
    printf("[green_finder -> green_finder_periodic()] Updated safe confidence: %d\n", safe_confidence);

    const float moveDistance = fminf(maxDistance, 0.2f * safe_confidence);
    const float attractive_adj = compute_attractive_adjustment(color_threshold);

    // Debug: Log attractive adjustment
    printf("[green_finder -> green_finder_periodic()] Attractive adjustment: %.2f\n", attractive_adj);

    // Increment the waypoint update counter
    waypoint_update_counter++;
    printf("percentage of green pixels: %d\n", color_count);

    // State Machine:
    switch (navigation_state) {
        case GREEN_SAFE:
            printf("[green_finder -> green_finder_periodic()] State: GREEN_SAFE\n");
            increase_nav_heading(attractive_adj);
            moveWaypointForward(WP_TRAJECTORY, 1.0f * moveDistance);
                // Check if the green pixel count is below the unsafe threshold
            if (color_count < unsafe_color_threshold) {
              printf("[green_finder -> green_finder_periodic()] Very unsafe! Stopping forward motion and rotating 30 degrees.\n");
              increase_nav_heading(30.0f); // Rotate 30 degrees
              moveWaypointForward(WP_TRAJECTORY, 0.0f); // Stop forward motion
              safe_confidence = 0; // Reset safe confidence
              navigation_state = GREEN_LOST;
              return; // Stop further processing in this cycle
            }
            waypoint_update_counter = 0; // Reset counter when a waypoint is updated
            if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
                navigation_state = OUT_OF_BOUNDS;
            } else if (safe_confidence == 0) {
                navigation_state = GREEN_LOST;
            } else {
                moveWaypointForward(WP_GOAL, moveDistance);
                moveWaypointForward(WP_RETREAT, -moveDistance);
                waypoint_update_counter = 0; // Reset counter when a waypoint is updated
            }
            break;

        case GREEN_LOST:
            printf("[green_finder -> green_finder_periodic()] State: GREEN_LOST\n");
            waypoint_move_here_2d(WP_GOAL);
            waypoint_move_here_2d(WP_RETREAT);
            waypoint_move_here_2d(WP_TRAJECTORY);
            if (fabsf(attractive_adj) < 10.0f || safe_confidence < 5) {
                increase_nav_heading(20.0f);
            } else {
                increase_nav_heading(attractive_adj);
            }
            if (safe_confidence >= 2) navigation_state = SEARCH_FOR_GREEN_HEADING;
            break;

        case SEARCH_FOR_GREEN_HEADING:
            printf("[green_finder -> green_finder_periodic()] State: SEARCH_FOR_GREEN_HEADING\n");
            waypoint_update_counter++;
            increase_nav_heading(attractive_adj);
            if (safe_confidence >= 2) {
                navigation_state = GREEN_SAFE;
            } else if (waypoint_update_counter > 5) {
                waypoint_update_counter = 0;
                navigation_state = GREEN_LOST;
            }
            break;

        case OUT_OF_BOUNDS:
            printf("[green_finder -> green_finder_periodic()] State: OUT_OF_BOUNDS\n");
            if (fabsf(attractive_adj) < 13.0f || safe_confidence < 2) {
                if (waypoint_update_counter > 10) {
                    printf("[green_finder -> green_finder_periodic()] Turning aggressively (120 degrees)\n");
                    increase_nav_heading(120.0f);
                    waypoint_update_counter = 0;
                } else {
                    printf("[green_finder -> green_finder_periodic()] Turning moderately (75 degrees)\n");
                    increase_nav_heading(90.0f);
                }
            } else {
                waypoint_update_counter++;
                increase_nav_heading(attractive_adj);
            }
            moveWaypointForward(WP_TRAJECTORY, 0.5f);
            moveWaypointForward(WP_RETREAT, -1.0f);
            waypoint_update_counter = 0;
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

  // Compute the magnitude of attraction based on green pixel count
  float attractive_magnitude = ((color_count - color_threshold) / (float) color_threshold);

  // Compute the offset from the image center (horizontal and vertical)
  float image_center_x = front_camera.output_size.w * 0.5f;
  float image_center_y = front_camera.output_size.h * 0.5f;
  float offset_x = (green_center_x - image_center_x) / image_center_x; // Horizontal offset
  float offset_y = (green_center_y - image_center_y) / image_center_y; // Vertical offset

  // Apply linear scaling to prioritize the center, but still consider top and bottom
  float weighted_offset_x = 1.0f - fabsf(offset_x); // Higher weight for closer to center horizontally
  float weighted_offset_y = 0.5f * (1.0f - fabsf(offset_y)); // Half the weight for vertical offset

  // Combine horizontal and vertical offsets with weights
  float combined_offset = (weighted_offset_x + weighted_offset_y) / 1.5f; // Normalize weights

  // Compute the final attractive adjustment
  return k_attr * attractive_magnitude * combined_offset;
}

// Provides a fallback increment when the computed attractive adjustment is near zero.
float fallback_increment_if_no_attraction(float attractive_adj)
{
  return (fabsf(attractive_adj) < 1e-3f) ? 90.0f : attractive_adj;
}

// Adjusts the navigation heading by a given increment (in degrees),
// but clamps the change to avoid excessive spinning.
void increase_nav_heading(float incrementDegrees)
{
  float current_heading = stateGetNedToBodyEulers_f()->psi;  // current heading in radians
  // Convert desired increment to radians.
  float desired_change = RadOfDeg(incrementDegrees);
  // Increase the maximum heading change (e.g., 30 degrees per update).
  float max_heading_change = RadOfDeg(200.0f); // 
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
