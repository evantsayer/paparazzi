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

 #include <pthread.h>
 #include "modules/everything_avoider_pf/everything_avoider_pf.h"
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
 #include <sys/stat.h>
 #include <sys/types.h>
 #include "onnxruntime_c_api.h"
 #include "lib/vision/image.h"  // Defines struct image_t

 #define NAV_C
 
 #define EVERYTHING_AVOIDER_VERBOSE TRUE
 
 #define PRINT(string, ...) fprintf(stderr, "[everything_avoider_pf->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
 #if EVERYTHING_AVOIDER_VERBOSE
 #define VERBOSE_PRINT PRINT
 #else
 #define VERBOSE_PRINT(...)
 #endif

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
 

 static void save_scaled_ppm(const float* buffer, int w, int h, const char* filename, bool is_normalized);

 /* ---------------- Global Variables for ONNX Runtime ---------------- */
 static const OrtApi* g_ort = NULL;
 static OrtEnv* g_env = NULL;
 static OrtSession* g_session = NULL;
 static OrtSessionOptions* g_session_options = NULL;

 // Add new globals for frame dimensions
 static int stored_width = 0;
 static int stored_height = 0;
 
    // ---------------- Global Variables ----------------

    // Mutex to protect access to latest_frame
    static pthread_mutex_t video_frame_mutex;

    // Pointer to the most recent camera frame (in UYVY format)
    static struct image_t *latest_frame = NULL;

    // (Other global variables and enums remain unchanged...)

    // ---------------- Video Callback Function ----------------

    /*
    * video_callback()
    * Called whenever a new video frame is available.
    */
    static struct image_t *video_callback(struct image_t *img, unsigned char id) {
        (void)id;
        // Log the incoming frame dimensions
        //VERBOSE_PRINT("video_callback: Received frame %p with dimensions %d x %d\n",
                    //(void*)img, img->w, img->h);
        pthread_mutex_lock(&video_frame_mutex);
        latest_frame = img;
        // Store the dimensions as soon as the frame arrives.
        stored_width = img->w;
        stored_height = img->h;
        pthread_mutex_unlock(&video_frame_mutex);
        return img;
    }


    /*
    * init_video_callback()
    * Registers the video callback with the front camera.
    */
    static void init_video_callback(void) {
        cv_add_to_device(&front_camera, video_callback, 5, 0); // 20 FPS, id = 0
    }
 
 /* ---------------- ONNX Model Loading ---------------- */
 /*
  * load_depth_model()
  * Loads the ONNX model from the specified file path.
  */
 void load_depth_model(const char* model_path) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "depth_model", &g_env) != ORT_OK) {
      VERBOSE_PRINT("Failed to create ONNX environment\n");
      exit(1);
    }
    if (g_ort->CreateSessionOptions(&g_session_options) != ORT_OK) {
      VERBOSE_PRINT("Failed to create ONNX session options\n");
      exit(1);
    }
    if (g_ort->CreateSession(g_env, model_path, g_session_options, &g_session) != ORT_OK) {
      VERBOSE_PRINT("Failed to create ONNX session\n");
      exit(1);
    }
 }
 
 /*
 * get_camera_image_normalized()
 * Converts the latest front-camera image (in UYVY format) into a normalized float tensor.
 */
 void get_camera_image_normalized(float* buffer) {
    const int width_out = 60;
    const int height_out = 130;
    const int output_elements = 1 * 3 * height_out * width_out;

    pthread_mutex_lock(&video_frame_mutex);
    struct image_t *frame = latest_frame;
    // Copy stored dimensions to local variables
    int width_in = stored_width;
    int height_in = stored_height;
    pthread_mutex_unlock(&video_frame_mutex);

    // Debug print using the stored dimensions
    VERBOSE_PRINT("get_camera_image_normalized: Using stored dimensions: width=%d, height=%d\n", width_in, height_in);

    if (frame == NULL || width_in == 0 || height_in == 0) {
        VERBOSE_PRINT("No valid camera frame available\n");
        for (int i = 0; i < output_elements; i++) {
            buffer[i] = 0.0f;
        }
        return;
    }

    int n_pixels = width_in * height_in;
    uint8_t* gray = (uint8_t*)malloc(n_pixels * sizeof(uint8_t));
    if (gray == NULL) {
        for (int i = 0; i < output_elements; i++) {
            buffer[i] = 0.0f;
        }
        return;
    }

    uint8_t* data = (uint8_t*)frame->buf;
    int group_count = n_pixels / 2;
    for (int i = 0; i < group_count; i++) {
        int base = i * 4;
        gray[2 * i]     = data[base + 1]; // Y for first pixel
        gray[2 * i + 1] = data[base + 3]; // Y for second pixel
    }

    // Optionally, save the raw image for debugging (converting 1-channel to 3-channel)
    float* raw_rgb = (float*)malloc(n_pixels * 3 * sizeof(float));
    if (raw_rgb != NULL) {
        for (int i = 0; i < n_pixels; i++) {
            raw_rgb[i * 3]     = (float)gray[i];
            raw_rgb[i * 3 + 1] = (float)gray[i];
            raw_rgb[i * 3 + 2] = (float)gray[i];
        }
        save_scaled_ppm(raw_rgb, width_in, height_in, "raw_image.ppm", false);
        free(raw_rgb);
    }

    // Downscale and normalize image
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
    int height_out = 130;
    int width_out = 60;
    // Updated dimensions: [batch, height, width, channels] -> [1, 130, 60, 3]
    int64_t input_dims[4] = {1, height_out, width_out, 3};
    size_t input_tensor_size = 1 * height_out * width_out * 3;  // total elements

    OrtMemoryInfo* memory_info = NULL;
    if (g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info) != ORT_OK) {
        VERBOSE_PRINT("Failed to create CPU memory info\n");
        exit(1);
    }

    OrtValue* input_tensor = NULL;
    if (g_ort->CreateTensorWithDataAsOrtValue(memory_info, (void*)input_tensor_data,
            input_tensor_size * sizeof(float), input_dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor) != ORT_OK) {
        VERBOSE_PRINT("Failed to create input tensor\n");
        exit(1);
    }

    const char* input_names[] = {"input"};
    const char* output_names[] = {"conv2d"};

    OrtValue* output_tensor = NULL;
    OrtStatus* status = g_ort->Run(g_session, NULL, input_names,
                          (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor);
    if (status != NULL) {
        const char* error_msg = g_ort->GetErrorMessage(status);
        VERBOSE_PRINT("Failed to run inference: %s\n", error_msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }

    float* out;
    if (g_ort->GetTensorMutableData(output_tensor, (void**)&out) != ORT_OK) {
        VERBOSE_PRINT("Failed to get output tensor data\n");
        exit(1);
    }
    size_t output_tensor_size = 1 * 1 * height_out * width_out;
    memcpy(output_tensor_data, out, output_tensor_size * sizeof(float));

    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
    VERBOSE_PRINT("Finished Inference\n");
  }


 
 /*
  * save_depth_map_to_ppm
  *
  * Saves a normalized depth map (values in [0,1]) with dimensions (width x height)
  * to a PPM file (P3 format) in the "depth_maps" folder under the module directory.
  * Each call will save a file with a unique filename.
  */
  void save_depth_map_to_ppm(const float* depth, int width, int height) {
      VERBOSE_PRINT("Saving image....\n");
      static int frame_counter = 0;
      char folder[] = "sw/airborne/modules/everything_avoider_pf/depth_maps";
      char filename[256];
  
      // Create folder if it doesn't exist.
      struct stat st = {0};
      if (stat(folder, &st) == -1) {
          if(mkdir(folder, 0700) == 0) {
              VERBOSE_PRINT("Created folder: %s\n", folder);
          } else {
              VERBOSE_PRINT("Error creating folder %s\n", folder);
          }
      }
      
      // Generate filename with frame counter.
      snprintf(filename, sizeof(filename), "%s/depth_map_%04d.ppm", folder, frame_counter++);
      
      FILE* fp = fopen(filename, "w");
      if (!fp) {
          VERBOSE_PRINT("Error opening file %s for writing\n", filename);
          return;
      }
      // Write PPM header
      fprintf(fp, "P3\n%d %d\n255\n", width, height);
      
      // Write pixel data: convert normalized float to 0-255 integer.
      for (int i = 0; i < width * height; i++) {
          int val = (int)(depth[i] * 255);
          if (val > 255) val = 255;
          if (val < 0)   val = 0;
          fprintf(fp, "%d %d %d ", val, val, val);
          if ((i + 1) % width == 0)
              fprintf(fp, "\n");
      }
      fclose(fp);
      VERBOSE_PRINT("Saved depth map to %s\n", filename);
  }

  /* 
  * Adjusted save_scaled_ppm function
  * If is_normalized is true, each float value is assumed to be in [0,1] and scaled by 255.
  * If false, the values are assumed to already be in [0,255] (raw data).
  */
  static void save_scaled_ppm(const float* buffer, int w, int h, const char* filename, bool is_normalized) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        VERBOSE_PRINT("Error opening %s\n", filename);
        return;
    }
    // P3 is the ASCII PPM format with max color 255.
    fprintf(fp, "P3\n%d %d\n255\n", w, h);
    int index = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float r = buffer[index++];
            float g = buffer[index++];
            float b = buffer[index++];
            int R, G, B;
            if (is_normalized) {
                R = (int)(r * 255.0f);
                G = (int)(g * 255.0f);
                B = (int)(b * 255.0f);
            } else {
                R = (int)r;
                G = (int)g;
                B = (int)b;
            }
            // Clamp to valid [0,255] range.
            if (R > 255) R = 255;
            if (G > 255) G = 255;
            if (B > 255) B = 255;
            fprintf(fp, "%d %d %d ", R, G, B);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    VERBOSE_PRINT("Saved image to %s (is_normalized: %s)\n", filename, is_normalized ? "true" : "false");
  }


 
 /* ---------------- Depth Map Processing ---------------- */
 /*
  * process_depth_map()
  * Captures the current camera image, runs ONNX inference to produce a depth map,
  * and processes the depth map to update 'red_count' and 'obstacle_center_x'.
  * Pixels with a depth value > 0.8 are considered obstacles.
  */
 void process_depth_map(void) {
    int width_target = 60;
    int height_target = 130;
    
    size_t input_size = 1 * 3 * height_target * width_target;
    float* input_buffer = (float*)malloc(input_size * sizeof(float));
    size_t output_size = 1 * 1 * height_target * width_target;
    float* output_buffer = (float*)malloc(output_size * sizeof(float));
  
    get_camera_image_normalized(input_buffer);

    VERBOSE_PRINT("Running Inference\n");

    run_depth_inference(input_buffer, output_buffer);
 
    VERBOSE_PRINT("DEBUG: Saving depth map...\n");
    fflush(stdout);
    save_depth_map_to_ppm(output_buffer, 60, 130);
 
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
  VERBOSE_PRINT("Beginning Periodic\n");
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
    load_depth_model("sw/airborne/modules/everything_avoider_pf/depth_cnn_model.onnx");

    // Initialize the mutex for video frame access
    pthread_mutex_init(&video_frame_mutex, NULL);

    init_video_callback();  // Register video callback to update 'latest_frame'
    navigation_state = SAFE;
    obstacle_free_confidence = 0;
    VERBOSE_PRINT("Module initialization complete.\n");
 }