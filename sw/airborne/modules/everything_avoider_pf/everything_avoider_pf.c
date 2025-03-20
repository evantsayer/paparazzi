/*
 * everything_avoider_pf.c
 *
 * This updated module integrates an ONNX-based neural network depth model
 * (exported from PyTorch as "depth_model.onnx") to compute a depth map from the
 * drone's front camera. The module obtains the latest video frame via a video
 * callback, converts the raw UYVY image into a normalized float tensor with shape
 * [1, 3, 130, 60] (batch, channels, height, width) – here, both dimensions are the
 * original dimensions divided by 4 – runs inference, and then processes the depth map
 * to detect obstacles based on high-valued (red) pixels. The resulting obstacle information
 * is used in a potential fields algorithm to adjust the drone's navigation.
 *
 * Note: Ensure that ONNX Runtime and the Paparazzi video libraries are properly linked.
 */

 #include "modules/everything_avoider_pf/everything_avoider_pf.h" // Header file (name kept for consistency)
 #include "firmwares/rotorcraft/navigation.h"
 #include "generated/airframe.h"
 #include "state.h"
 #include "modules/core/abi.h"
 #include "generated/flight_plan.h"
 
 #include <time.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <math.h>
 
 /* ONNX Runtime header */
 #include "onnxruntime_c_api.h"
 
 /* For video handling: include the image structure definition */
 #include "lib/vision/image.h"  // Defines struct image_t
 
 /* ---------------- Global Variables for Potential Fields ---------------- */
 float k_rep = 40.0f;
 float maxDistance = 1.50f;
 float oa_color_count_frac = 0.15f; // Now represents the threshold fraction for red (obstacle) pixel detection
 
 volatile int32_t red_count = 0;         // Total count of "red" (obstacle) pixels from depth map
 volatile int16_t obstacle_center_x = 0;   // Average x-coordinate of detected obstacle pixels
 int16_t obstacle_free_confidence = 0;
 const int16_t max_safe_confidence = 10;
 
 enum navigation_state_t {
   SAFE,
   OBSTACLE_FOUND,
   SEARCH_FOR_SAFE_HEADING,
   OUT_OF_BOUNDS
 };
 enum navigation_state_t navigation_state = SAFE;
 
 /* ---------------- Global Variables for ONNX Runtime ---------------- */
 static const OrtApi* g_ort = NULL;
 static OrtEnv* g_env = NULL;
 static OrtSession* g_session = NULL;
 static OrtSessionOptions* g_session_options = NULL;
 
 /* ---------------- Global Variable for Video Frame ---------------- */
 // Pointer to the most recent camera frame (in UYVY format)
 static struct image_t *latest_frame = NULL;
 
 /* ---------------- Video Callback Function ---------------- */
 /*
  * video_callback()
  * Called whenever a new video frame is available.
  * The expected function signature is:
  *   struct image_t *func(struct image_t *img, unsigned char id);
  * We store the pointer to the latest frame and return the image.
  */
 static struct image_t *video_callback(struct image_t *img, unsigned char id) {
     (void)id;  // Unused parameter
     latest_frame = img;
     return img;  // Pass the frame along for further processing if needed.
 }
 
 /*
  * init_video_callback()
  * Registers the video callback with the front camera.
  */
 static void init_video_callback(void) {
     // cv_add_to_device expects a function with signature: 
     //   struct image_t *(*cv_function)(struct image_t *, unsigned char)
     cv_add_to_device(&front_camera, video_callback, 20, 0); // 20 FPS, id = 0
 }
 
 /* ---------------- ONNX Model Loading ---------------- */
 /*
  * load_depth_model()
  * Loads the ONNX model from the specified file path.
  */
 void load_depth_model(const char* model_path) {
   g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
   if (g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "depth_model", &g_env) != ORT_OK) {
     fprintf(stderr, "Failed to create ONNX environment\n");
     exit(1);
   }
   if (g_ort->CreateSessionOptions(&g_session_options) != ORT_OK) {
     fprintf(stderr, "Failed to create ONNX session options\n");
     exit(1);
   }
   if (g_ort->CreateSession(g_env, model_path, g_session_options, &g_session) != ORT_OK) {
     fprintf(stderr, "Failed to create ONNX session\n");
     exit(1);
   }
 }
 
 /* ---------------- Camera Image Acquisition ---------------- */
 /*
  * get_camera_image_normalized()
  * Converts the latest front-camera image (in UYVY format) into a normalized float tensor.
  * The output tensor has shape: [1, 3, 130, 60] (batch, channels, height, width),
  * where both dimensions are the original dimensions divided by 4 (e.g., from 240×520 to 60×130).
  * We extract the luminance (Y) channel and then downscale the image using nearest-neighbor interpolation.
  */
 void get_camera_image_normalized(float* buffer) {
     const int width_out = 60;   // target width: original width / 4 (e.g., 240/4)
     const int height_out = 130; // target height: original height / 4 (e.g., 520/4)
     const int output_elements = 1 * 3 * height_out * width_out;
 
     if (latest_frame == NULL) {
         for (int i = 0; i < output_elements; i++) {
             buffer[i] = 0.0f;
         }
         return;
     }
     
     int width_in = latest_frame->w;    // e.g., 240
     int height_in = latest_frame->h;   // e.g., 520
     int n_pixels = width_in * height_in;
     
     uint8_t* gray = (uint8_t*)malloc(n_pixels * sizeof(uint8_t));
     if (gray == NULL) {
         for (int i = 0; i < output_elements; i++) {
             buffer[i] = 0.0f;
         }
         return;
     }
     
     // Cast the buffer pointer to uint8_t*
     uint8_t* data = (uint8_t*)latest_frame->buf;
     int group_count = n_pixels / 2;
     for (int i = 0; i < group_count; i++) {
         int base = i * 4;
         gray[2 * i]     = data[base + 1]; // Y for first pixel
         gray[2 * i + 1] = data[base + 3]; // Y for second pixel
     }
     
     // Downscale using nearest-neighbor interpolation.
     float scale_x = (float)width_in / width_out;
     float scale_y = (float)height_in / height_out;
     
     int out_index = 0;
     for (int row = 0; row < height_out; row++) {
         int in_y = (int)floor(row * scale_y);
         if (in_y >= height_in) in_y = height_in - 1;
         for (int col = 0; col < width_out; col++) {
             int in_x = (int)floor(col * scale_x);
             if (in_x >= width_in) in_x = width_in - 1;
             uint8_t pixel_val = gray[in_y * width_in + in_x];
             float norm = pixel_val / 255.0f;
             // Replicate the normalized value into 3 channels (R, G, B).
             buffer[out_index++] = norm;
             buffer[out_index++] = norm;
             buffer[out_index++] = norm;
         }
     }
     
     free(gray);
 }
 
 /* ---------------- ONNX Inference ---------------- */
 /*
  * run_depth_inference()
  * Runs the ONNX model on the normalized input tensor and writes the output depth map
  * to output_tensor_data. Assumes input shape [1, 3, 130, 60] and output shape [1, 1, 130, 60].
  */
 void run_depth_inference(const float* input_tensor_data, float* output_tensor_data) {
   int width_out = 60;
   int height_out = 130;
   int64_t input_dims[4] = {1, 3, height_out, width_out};
   size_t input_tensor_size = 1 * 3 * height_out * width_out;
 
   OrtMemoryInfo* memory_info = NULL;
   if (g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info) != ORT_OK) {
     fprintf(stderr, "Failed to create CPU memory info\n");
     exit(1);
   }
 
   OrtValue* input_tensor = NULL;
   if (g_ort->CreateTensorWithDataAsOrtValue(memory_info, (void*)input_tensor_data,
       input_tensor_size * sizeof(float), input_dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
       &input_tensor) != ORT_OK) {
     fprintf(stderr, "Failed to create input tensor\n");
     exit(1);
   }
 
   const char* input_names[] = {"input"};
   const char* output_names[] = {"output"};
 
   OrtValue* output_tensor = NULL;
   if (g_ort->Run(g_session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor) != ORT_OK) {
     fprintf(stderr, "Failed to run inference\n");
     exit(1);
   }
 
   float* out;
   if (g_ort->GetTensorMutableData(output_tensor, (void**)&out) != ORT_OK) {
     fprintf(stderr, "Failed to get output tensor data\n");
     exit(1);
   }
   size_t output_tensor_size = 1 * 1 * height_out * width_out;
   memcpy(output_tensor_data, out, output_tensor_size * sizeof(float));
 
   g_ort->ReleaseValue(output_tensor);
   g_ort->ReleaseValue(input_tensor);
   g_ort->ReleaseMemoryInfo(memory_info);
 }
 
 /* ---------------- Depth Map Processing ---------------- */
 /*
  * process_depth_map()
  * Captures the current camera image, runs ONNX inference to produce a depth map,
  * and processes the depth map to update 'red_count' and 'obstacle_center_x'.
  *
  * In this example, pixels with a depth value > 0.8 are considered obstacles.
  */
 void process_depth_map(void) {
   int width_target = 60;
   int height_target = 130;
   
   size_t input_size = 1 * 3 * height_target * width_target;
   float* input_buffer = (float*)malloc(input_size * sizeof(float));
   size_t output_size = 1 * 1 * height_target * width_target;
   float* output_buffer = (float*)malloc(output_size * sizeof(float));
 
   get_camera_image_normalized(input_buffer);
   run_depth_inference(input_buffer, output_buffer);
 
   int count = 0;
   long sum_x = 0;
   float threshold = 0.8f;  // Tunable threshold for obstacle detection
   for (int y = 0; y < height_target; y++) {
       for (int x = 0; x < width_target; x++) {
           int index = y * width_target + x;
           if (output_buffer[index] > threshold) {
               count++;
               sum_x += x;
           }
       }
   }
   red_count = count;
   if (count > 0)
       obstacle_center_x = (int16_t)(sum_x / count);
   else
       obstacle_center_x = width_target / 2;
 
   free(input_buffer);
   free(output_buffer);
 }
 
 /* ---------------- Helper Functions (unchanged) ---------------- */
 float compute_repulsive_adjustment(int32_t red_threshold) {
   if (red_count < red_threshold)
     return 0.0f;
   float repulsive_magnitude = ((red_count - red_threshold) * (1.0f / red_threshold));
   float offset = (front_camera.output_size.w * 0.5f - obstacle_center_x) *
                  (1.0f / (front_camera.output_size.w * 0.5f));
   return k_rep * repulsive_magnitude * offset;
 }
 
 float fallback_increment_if_no_repulsion(float repulsive_adj) {
   return (fabsf(repulsive_adj) < 1e-3f) ? 10.0f : repulsive_adj;
 }
 
 void increase_nav_heading(float incrementDegrees) {
   float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
   FLOAT_ANGLE_NORMALIZE(new_heading);
   nav.heading = new_heading;
 }
 
 void calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters) {
   const struct FloatEulers* eulers = stateGetNedToBodyEulers_f();
   const struct EnuCoor_i* pos = stateGetPositionEnu_i();
   float sin_h = sinf(eulers->psi);
   float cos_h = cosf(eulers->psi);
   new_coor->x = pos->x + POS_BFP_OF_REAL(sin_h * distanceMeters);
   new_coor->y = pos->y + POS_BFP_OF_REAL(cos_h * distanceMeters);
 }
 
 void moveWaypoint(uint8_t waypoint, const struct EnuCoor_i *new_coor) {
   waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
 }
 
 void moveWaypointForward(uint8_t waypoint, float distanceMeters) {
   struct EnuCoor_i new_coor;
   calculateForwards(&new_coor, distanceMeters);
   moveWaypoint(waypoint, &new_coor);
 }
 
 /* ---------------- Main Periodic Function ---------------- */
 /*
  * everything_avoider_pf_periodic()
  * Called periodically (e.g., at 4Hz) to:
  *   1. Process the depth map (via ONNX inference) to update obstacle information.
  *   2. Adjust navigation using the potential fields state machine.
  */
 void everything_avoider_pf_periodic(void) {
   if (!autopilot_in_flight())
     return;
 
   process_depth_map();
 
   int width = front_camera.output_size.w;
   int height = front_camera.output_size.h;
   int32_t total_pixels = width * height;
   int32_t red_threshold = (int32_t)(oa_color_count_frac * total_pixels);
 
   if (red_count < red_threshold) {
     if (obstacle_free_confidence < max_safe_confidence)
       obstacle_free_confidence++;
   } else {
     obstacle_free_confidence = (obstacle_free_confidence > 2) ? obstacle_free_confidence - 2 : 0;
   }
 
   float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);
   float repulsive_adj = compute_repulsive_adjustment(red_threshold);
 
   switch (navigation_state) {
     case SAFE:
       increase_nav_heading(repulsive_adj);
       moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);
       if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY)))
           navigation_state = OUT_OF_BOUNDS;
       else if (obstacle_free_confidence == 0)
           navigation_state = OBSTACLE_FOUND;
       else {
           moveWaypointForward(WP_GOAL, moveDistance);
           moveWaypointForward(WP_RETREAT, -moveDistance);
       }
       break;
 
     case OBSTACLE_FOUND:
       waypoint_move_here_2d(WP_GOAL);
       waypoint_move_here_2d(WP_RETREAT);
       waypoint_move_here_2d(WP_TRAJECTORY);
       increase_nav_heading(repulsive_adj);
       if (obstacle_free_confidence >= 2)
           navigation_state = SEARCH_FOR_SAFE_HEADING;
       break;
 
     case SEARCH_FOR_SAFE_HEADING:
       increase_nav_heading(repulsive_adj);
       if (obstacle_free_confidence >= 2)
           navigation_state = SAFE;
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
 
 /* ---------------- Module Initialization ---------------- */
 /*
  * everything_avoider_pf_init()
  * Initializes the module by:
  *   - Loading the ONNX model.
  *   - Registering the video callback to capture front-camera images.
  *   - Setting the initial state.
  */
 void everything_avoider_pf_init(void) {
   srand((unsigned)time(NULL));
   load_depth_model("sw/airborne/modules/everything_avoider_pf/depth_cnn_model.onnx");  // Ensure your exported ONNX model is available at this path
   init_video_callback();                 // Register video callback to update 'latest_frame'
   navigation_state = SAFE;
   obstacle_free_confidence = 0;
 }
 