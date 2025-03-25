/*
 * everything_avoider_pf.c
 *
 * Updated module using a new ONNX-based model.
 * The new model resizes the image to 60x130 then crops off the left part,
 * yielding a final input tensor of shape [1, 3, 130, 40] (interpreted as [1, 130, 40, 3]
 * by the ONNX model). The model outputs a single float in [0,1] indicating a normalized
 * desired direction (0 = far left, 0.5 = straight ahead, 1 = far right).
 *
 * The NN output is used to update the drone’s heading and waypoints.
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
 
 /* ---------------- Tunable Parameters Definitions ---------------- */
 /* These definitions resolve the undefined reference errors from settings */
 float maxAngleDegrees = 30.0f;  // Maximum heading adjustment in degrees
 float moveDistance = 1.5f;      // Forward move distance in SAFE state
 float fallbackDistance = 1.0f;  // Forward move distance in OUT_OF_BOUNDS state
 
 /* ---------------- Global Variables for Drone Navigation ---------------- */
 // Use the typedef from the header
 navigation_state_t navigation_state = SAFE;
 
 /* ---------------- Global Variables for ONNX Runtime ---------------- */
 static const OrtApi* g_ort = NULL;
 static OrtEnv* g_env = NULL;
 static OrtSession* g_session = NULL;
 static OrtSessionOptions* g_session_options = NULL;
 
 /* Global variables for frame dimensions */
 static int stored_width = 0;
 static int stored_height = 0;
 
 /* ---------------- Global Variables for Video Handling ---------------- */
 static pthread_mutex_t video_frame_mutex;
 static struct image_t *latest_frame = NULL;
 
 /* ---------------- Video Callback Function ---------------- */
 static struct image_t *video_callback(struct image_t *img, unsigned char id) {
     (void)id;
     pthread_mutex_lock(&video_frame_mutex);
     latest_frame = img;
     stored_width = img->w;
     stored_height = img->h;
     pthread_mutex_unlock(&video_frame_mutex);
     //VERBOSE_PRINT("Received frame %p with dimensions %d x %d\n", img, img->w, img->h);
     return img;
 }
 
 static void init_video_callback(void) {
     cv_add_to_device(&front_camera, video_callback, 5, 0); // 20 FPS, id = 0
     VERBOSE_PRINT("Video callback registered for front camera.\n");
 }
 
 /* ---------------- ONNX Model Loading ---------------- */
 static void load_depth_model(const char* model_path) {
     g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
     if (g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "depth_model", &g_env) != ORT_OK) {
       VERBOSE_PRINT("Failed to create ONNX environment\n");
       exit(1);
     }
     VERBOSE_PRINT("ONNX environment created.\n");
     
     if (g_ort->CreateSessionOptions(&g_session_options) != ORT_OK) {
       VERBOSE_PRINT("Failed to create ONNX session options\n");
       exit(1);
     }
     VERBOSE_PRINT("ONNX session options created.\n");
     
     if (g_ort->CreateSession(g_env, model_path, g_session_options, &g_session) != ORT_OK) {
       VERBOSE_PRINT("Failed to create ONNX session\n");
       exit(1);
     }
     VERBOSE_PRINT("ONNX session created using model at %s\n", model_path);
 }
 
    /* ---------------- Modified Image Normalization ---------------- */
    /*
    * get_camera_image_normalized()
    *
    * Converts the latest front-camera image (in UYVY format) into a normalized float tensor.
    * New pipeline:
    *   1. Resize image to 60x130.
    *   2. Crop off the left 20 columns (from the 60-column image) to yield a 40x130 image.
    *
    * The output tensor has shape [1, 3, NN_HEIGHT, NN_FINAL_WIDTH].
    */
    static void get_camera_image_normalized(float* final_buffer) {
        const int intermediate_width = NN_INTERMEDIATE_WIDTH; // 60
        const int height_out = NN_HEIGHT;                     // 130
        const int final_width = NN_FINAL_WIDTH;               // 40
        const int intermediate_elements = 1 * 3 * height_out * intermediate_width;
        const int final_elements = 1 * 3 * height_out * final_width;
        
        pthread_mutex_lock(&video_frame_mutex);
        struct image_t *frame = latest_frame;
        int width_in = stored_width;
        int height_in = stored_height;
        pthread_mutex_unlock(&video_frame_mutex);

        VERBOSE_PRINT("get_camera_image_normalized: Using input dimensions %d x %d\n", width_in, height_in);
        
        if (frame == NULL || frame->buf == NULL) {
            VERBOSE_PRINT("Error: No valid frame or frame->buf is NULL\n");
            for (int i = 0; i < final_elements; i++) {
                final_buffer[i] = 0.0f;
            }
            return;
        }
        
        // For UYVY, expected buffer size is width_in * height_in * 2 bytes.
        int expected_buffer_size = width_in * height_in * 2;
        VERBOSE_PRINT("Expected buffer size (in bytes): %d\n", expected_buffer_size);
        
        // Since our image_t doesn't have a buf_len, we assume the size is as expected.
        int actual_buffer_size = expected_buffer_size;
        
        // Cast frame->buf to uint8_t pointer.
        uint8_t *buf = (uint8_t *)frame->buf;
        
        // For debugging, only copy a small portion of the buffer (e.g., 1024 bytes)
        int debug_copy_size = 1024;
        if (expected_buffer_size < debug_copy_size)
            debug_copy_size = expected_buffer_size;
        
        uint8_t *local_buf = (uint8_t *)malloc(debug_copy_size);
        if (local_buf == NULL) {
            VERBOSE_PRINT("Failed to allocate local buffer for debug copy.\n");
            for (int i = 0; i < final_elements; i++) {
                final_buffer[i] = 0.0f;
            }
            return;
        }
        
        memcpy(local_buf, buf, debug_copy_size);
        VERBOSE_PRINT("Copied %d bytes from frame->buf to local buffer for debugging.\n", debug_copy_size);
        
        // Print the first 32 bytes of the local buffer.
        char hex_str[256] = {0};
        int print_len = 32;
        if (debug_copy_size < print_len)
            print_len = debug_copy_size;
        for (int i = 0; i < print_len; i++) {
            char temp[4];
            sprintf(temp, "%02x ", local_buf[i]);
            strcat(hex_str, temp);
        }
        VERBOSE_PRINT("First 32 bytes of local buffer: %s\n", hex_str);
        
        // Compute average values over the debug block (each group of 4 bytes represents two pixels: [U, Y, V, Y]).
        int num_groups = debug_copy_size / 4;
        unsigned long long sum_U = 0, sum_Y1 = 0, sum_V = 0, sum_Y2 = 0;
        for (int i = 0; i < num_groups; i++) {
            int base = i * 4;
            if (base + 3 >= debug_copy_size)
                break;
            sum_U  += local_buf[base];
            sum_Y1 += local_buf[base + 1];
            sum_V  += local_buf[base + 2];
            sum_Y2 += local_buf[base + 3];
        }
        double avg_U = sum_U / (double)num_groups;
        double avg_Y1 = sum_Y1 / (double)num_groups;
        double avg_V = sum_V / (double)num_groups;
        double avg_Y2 = sum_Y2 / (double)num_groups;
        VERBOSE_PRINT("Debug Averages over first %d groups - Avg U: %f, Avg Y1: %f, Avg V: %f, Avg Y2: %f\n",
                    num_groups, avg_U, avg_Y1, avg_V, avg_Y2);
        free(local_buf);

        // (If the debug copy works, you may then try processing the full buffer.)
        // If you still get a segfault when copying more than 1024 bytes,
        // it indicates that the frame->buf pointer isn’t valid for the full expected size.
        // At this point, you may need to check your camera configuration or driver.
        
        // ... (The rest of your processing: converting UYVY to grayscale, resizing, cropping, etc.)
    }



 /* ---------------- Modified ONNX Inference ---------------- */
 /*
  * run_depth_inference()
  *
  * Runs the ONNX model on the normalized input tensor and writes the output to output_tensor_data.
  * New input shape: [1, NN_HEIGHT, NN_FINAL_WIDTH, 3] i.e. [1, 130, 40, 3]
  * New output: a single float value representing the normalized direction.
  */
 static void run_depth_inference(const float* input_tensor_data, float* output_tensor_data) {
     int height_out = NN_HEIGHT;       // 130
     int final_width = NN_FINAL_WIDTH;   // 40
     int64_t input_dims[4] = {1, height_out, final_width, 3};
     size_t input_tensor_size = 1 * height_out * final_width * 3;
 
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
     const char* output_names[] = {"dense"};
 
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
     /* New model outputs a single float value. */
     output_tensor_data[0] = out[0];
 
     VERBOSE_PRINT("Inference completed. Output value: %f\n", out[0]);
 
     g_ort->ReleaseValue(output_tensor);
     g_ort->ReleaseValue(input_tensor);
     g_ort->ReleaseMemoryInfo(memory_info);
 }
 
 /* ---------------- New NN Output Processing ---------------- */
 /*
  * process_nn_output()
  *
  * Captures the current camera image, runs NN inference to produce a normalized direction,
  * and returns the output value.
  */
 static float process_nn_output(void) {
     int final_width = NN_FINAL_WIDTH; // 40
     int height_out = NN_HEIGHT;         // 130
     size_t input_size = 1 * 3 * height_out * final_width;
     float* input_buffer = (float*)malloc(input_size * sizeof(float));
     size_t output_size = 1;
     float* output_buffer = (float*)malloc(output_size * sizeof(float));
 
     VERBOSE_PRINT("Capturing and normalizing camera image...\n");
     get_camera_image_normalized(input_buffer);
     VERBOSE_PRINT("Image normalization complete.\n");
 
     VERBOSE_PRINT("Running ONNX inference...\n");
     run_depth_inference(input_buffer, output_buffer);
     float nn_direction = output_buffer[0];
     VERBOSE_PRINT("NN inference produced direction: %f\n", nn_direction);
 
     free(input_buffer);
     free(output_buffer);
     return nn_direction;
 }
 
 /* ---------------- Helper Functions ---------------- */
 /*
  * adjust_heading()
  *
  * Adjusts the drone's heading by the specified degree increment.
  */
 static void adjust_heading(float incrementDegrees) {
     float old_heading = stateGetNedToBodyEulers_f()->psi;
     float new_heading = old_heading + RadOfDeg(incrementDegrees);
     FLOAT_ANGLE_NORMALIZE(new_heading);
     nav.heading = new_heading;
     VERBOSE_PRINT("Heading adjusted from %f to %f (increment: %f degrees)\n", old_heading, new_heading, incrementDegrees);
 }
 
 /* ---------------- Additional Helper Functions ---------------- */
 void increase_nav_heading(float incrementDegrees) {
     float old_heading = stateGetNedToBodyEulers_f()->psi;
     float new_heading = old_heading + RadOfDeg(incrementDegrees);
     FLOAT_ANGLE_NORMALIZE(new_heading);
     nav.heading = new_heading;
     VERBOSE_PRINT("increase_nav_heading: Changed heading from %f to %f (increment: %f degrees)\n", old_heading, new_heading, incrementDegrees);
 }
   
 void calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters) {
     const struct FloatEulers* eulers = stateGetNedToBodyEulers_f();
     const struct EnuCoor_i* pos = stateGetPositionEnu_i();
     float sin_h = sinf(eulers->psi);
     float cos_h = cosf(eulers->psi);
     new_coor->x = pos->x + POS_BFP_OF_REAL(sin_h * distanceMeters);
     new_coor->y = pos->y + POS_BFP_OF_REAL(cos_h * distanceMeters);
     VERBOSE_PRINT("calculateForwards: New coordinates calculated: x=%d, y=%d\n", new_coor->x, new_coor->y);
 }
   
 void moveWaypoint(uint8_t waypoint, const struct EnuCoor_i *new_coor) {
     waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
     VERBOSE_PRINT("moveWaypoint: Moved waypoint %d to new coordinates: x=%d, y=%d\n", waypoint, new_coor->x, new_coor->y);
 }
   
 void moveWaypointForward(uint8_t waypoint, float distanceMeters) {
     struct EnuCoor_i new_coor;
     calculateForwards(&new_coor, distanceMeters);
     moveWaypoint(waypoint, &new_coor);
     VERBOSE_PRINT("moveWaypointForward: Waypoint %d moved forward by %f meters.\n", waypoint, distanceMeters);
 }
 
 /* ---------------- Modified Main Periodic Function ---------------- */
 /*
  * everything_avoider_pf_periodic()
  *
  * Called periodically to:
  *   1. Process the NN output to update the desired direction.
  *   2. Adjust navigation using a simplified state machine.
  */
 void everything_avoider_pf_periodic(void) {
     VERBOSE_PRINT("Periodic function started.\n");
     if (!autopilot_in_flight()) {
         VERBOSE_PRINT("Autopilot not in flight. Exiting periodic function.\n");
         return;
     }
 
     // Obtain NN output: a normalized value [0,1]
     float nn_direction = process_nn_output();
     VERBOSE_PRINT("NN direction output: %f\n", nn_direction);
 
     // Map NN output to a heading adjustment.
     float angle_adjustment = (nn_direction - 0.5f) * 2.0f * maxAngleDegrees;
     VERBOSE_PRINT("Computed heading adjustment: %f degrees\n", angle_adjustment);
 
     switch (navigation_state) {
       case SAFE:
           VERBOSE_PRINT("State: SAFE\n");
           adjust_heading(angle_adjustment);
           moveWaypointForward(WP_TRAJECTORY, moveDistance);
           moveWaypointForward(WP_GOAL, moveDistance);
           moveWaypointForward(WP_RETREAT, -moveDistance);
           if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
               navigation_state = OUT_OF_BOUNDS;
               VERBOSE_PRINT("Switching state to OUT_OF_BOUNDS\n");
           }
           break;
 
       case OUT_OF_BOUNDS:
           VERBOSE_PRINT("State: OUT_OF_BOUNDS\n");
           adjust_heading((nn_direction - 0.5f) * 2.0f * (maxAngleDegrees / 2.0f));
           moveWaypointForward(WP_TRAJECTORY, fallbackDistance);
           if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
               navigation_state = SAFE;
               VERBOSE_PRINT("Switching state back to SAFE\n");
           }
           break;
     }
 }
 
 /* ---------------- Module Initialization ---------------- */
 /*
  * everything_avoider_pf_init()
  *
  * Initializes the module by:
  *   - Loading the ONNX model.
  *   - Registering the video callback.
  *   - Setting the initial state.
  */
 void everything_avoider_pf_init(void) {
     srand((unsigned)time(NULL));
     VERBOSE_PRINT("Initializing everything_avoider_pf module...\n");
     load_depth_model("sw/airborne/modules/everything_avoider_pf/depth_cnn_model_epoch_15.onnx");
     pthread_mutex_init(&video_frame_mutex, NULL);
     init_video_callback();
     navigation_state = SAFE;
     VERBOSE_PRINT("Module initialization complete. Navigation state set to SAFE.\n");
 }
 