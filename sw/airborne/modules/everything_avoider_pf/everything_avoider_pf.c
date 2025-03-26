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
 float maxAngleDegrees = 15.0f;  // Maximum heading adjustment in degrees
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
        // Intermediate dimensions (before cropping)
        const int intermediate_width = NN_INTERMEDIATE_WIDTH; // e.g. 60
        const int height_out = NN_HEIGHT;                     // e.g. 130
        // Final dimensions after cropping (if desired)
        const int final_width = NN_FINAL_WIDTH;               // e.g. 40
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
        
        // For UYVY images (IMAGE_YUV422), the expected size is width_in * height_in * 2 bytes.
        int expected_buffer_size = width_in * height_in * 2;
        uint32_t actual_buffer_size = frame->buf_size;
        VERBOSE_PRINT("Expected buffer size (in bytes): %d, Actual buffer size: %d\n",
                    expected_buffer_size, actual_buffer_size);
        
        if (actual_buffer_size < expected_buffer_size) {
            VERBOSE_PRINT("Error: Actual buffer size is smaller than expected!\n");
            for (int i = 0; i < final_elements; i++) {
                final_buffer[i] = 0.0f;
            }
            return;
        }
        
        // Cast the buffer to a uint8_t pointer.
        uint8_t *buf = (uint8_t *)frame->buf;
        
        // Allocate a local copy of the full buffer.
        uint8_t *local_buf = (uint8_t *)malloc(expected_buffer_size);
        if (local_buf == NULL) {
            VERBOSE_PRINT("Failed to allocate local buffer for full copy.\n");
            for (int i = 0; i < final_elements; i++) {
                final_buffer[i] = 0.0f;
            }
            return;
        }
        memcpy(local_buf, buf, expected_buffer_size);
        VERBOSE_PRINT("Copied full buffer (%d bytes) from frame->buf to local buffer.\n", expected_buffer_size);
        
        // Debug: Print first 32 bytes from the local buffer.
        {
            char hex_str[256] = {0};
            int print_len = 32;
            if (print_len > expected_buffer_size)
                print_len = expected_buffer_size;
            for (int i = 0; i < print_len; i++) {
                char temp[4];
                sprintf(temp, "%02x ", local_buf[i]);
                strcat(hex_str, temp);
            }
            VERBOSE_PRINT("First 32 bytes of local buffer: %s\n", hex_str);
        }
        
        // Compute average values from the UYVY data.
        // Each 4-byte group represents two pixels: [U, Y, V, Y].
        int total_pixels = width_in * height_in;
        int num_groups = total_pixels / 2;
        unsigned long long sum_U = 0, sum_Y1 = 0, sum_V = 0, sum_Y2 = 0;
        for (int i = 0; i < num_groups; i++) {
            int base = i * 4;
            if (base + 3 >= expected_buffer_size)
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
        VERBOSE_PRINT("Full buffer Averages - Avg U: %f, Avg Y1: %f, Avg V: %f, Avg Y2: %f\n",
                    avg_U, avg_Y1, avg_V, avg_Y2);
        
        if (avg_Y1 < 10.0 && avg_Y2 < 10.0) {
            VERBOSE_PRINT("Warning: Both average Y values are very low (image may be underexposed or not in UYVY format).\n");
        }
        
        // Convert the full UYVY image to a grayscale image by extracting the Y values.
        int n_pixels = width_in * height_in;
        uint8_t *gray = (uint8_t*)malloc(n_pixels * sizeof(uint8_t));
        if (gray == NULL) {
            VERBOSE_PRINT("Failed to allocate gray buffer.\n");
            free(local_buf);
            for (int i = 0; i < final_elements; i++) {
                final_buffer[i] = 0.0f;
            }
            return;
        }
        for (int i = 0; i < num_groups; i++) {
            int base = i * 4;
            if (base + 3 >= expected_buffer_size)
                break;
            gray[2 * i]     = local_buf[base + 1];  // Y value from first pixel
            gray[2 * i + 1] = local_buf[base + 3];  // Y value from second pixel
        }
        VERBOSE_PRINT("Converted full UYVY buffer to grayscale image.\n");
        free(local_buf);
        
        // Resize the grayscale image to an intermediate resolution of NN_INTERMEDIATE_WIDTH x NN_HEIGHT.
        float* intermediate_buffer = (float*)malloc(intermediate_elements * sizeof(float));
        if (intermediate_buffer == NULL) {
            VERBOSE_PRINT("Failed to allocate intermediate buffer.\n");
            free(gray);
            for (int i = 0; i < final_elements; i++) {
                final_buffer[i] = 0.0f;
            }
            return;
        }
        float scale_x = (float)width_in / (float)NN_INTERMEDIATE_WIDTH;
        float scale_y = (float)height_in / (float)NN_HEIGHT;
        int out_index = 0;
        for (int row = 0; row < NN_HEIGHT; row++) {
            int in_y = (int)floor(row * scale_y);
            if (in_y >= height_in)
                in_y = height_in - 1;
            for (int col = 0; col < NN_INTERMEDIATE_WIDTH; col++) {
                int in_x = (int)floor(col * scale_x);
                if (in_x >= width_in)
                    in_x = width_in - 1;
                uint8_t pixel_val = gray[in_y * width_in + in_x];
                float norm = pixel_val / 255.0f;
                intermediate_buffer[out_index++] = norm;
                intermediate_buffer[out_index++] = norm;
                intermediate_buffer[out_index++] = norm;
            }
        }
        free(gray);
        VERBOSE_PRINT("Resized grayscale image to intermediate dimensions: %dx%d.\n", NN_INTERMEDIATE_WIDTH, NN_HEIGHT);
        
        // Crop the intermediate image to the final resolution.
        // For example, if cropping off the left part to obtain NN_FINAL_WIDTH columns.
        for (int row = 0; row < NN_HEIGHT; row++) {
            for (int col = 0; col < NN_FINAL_WIDTH; col++) {
                // Adjust the cropping offset as needed.
                int src_col = col + (NN_INTERMEDIATE_WIDTH - NN_FINAL_WIDTH); // cropping from left side
                int src_index = (row * NN_INTERMEDIATE_WIDTH + src_col) * 3;
                int dst_index = (row * NN_FINAL_WIDTH + col) * 3;
                final_buffer[dst_index]     = intermediate_buffer[src_index];
                final_buffer[dst_index + 1] = intermediate_buffer[src_index + 1];
                final_buffer[dst_index + 2] = intermediate_buffer[src_index + 2];
            }
        }
        free(intermediate_buffer);
        VERBOSE_PRINT("Cropped resized image to final dimensions: %dx%d.\n", NN_FINAL_WIDTH, NN_HEIGHT);
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

        // In SAFE mode, use the NN output to compute a heading adjustment.
        float angle_adjustment = (nn_direction - 0.5f) * 2.0f * maxAngleDegrees;
        VERBOSE_PRINT("Computed heading adjustment (SAFE): %f degrees\n", angle_adjustment);

        switch (navigation_state) {
            case SAFE:
                VERBOSE_PRINT("State: SAFE\n");
                adjust_heading(angle_adjustment);
                moveWaypointForward(WP_TRAJECTORY, moveDistance);
                moveWaypointForward(WP_GOAL, moveDistance);
                moveWaypointForward(WP_RETREAT, -moveDistance);
                // If the trajectory waypoint is out of bounds, switch to OUT_OF_BOUNDS state.
                if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
                    navigation_state = OUT_OF_BOUNDS;
                    VERBOSE_PRINT("Switching state to OUT_OF_BOUNDS\n");
                }
                break;

            case OUT_OF_BOUNDS:
                VERBOSE_PRINT("State: OUT_OF_BOUNDS\n");
                // In OUT_OF_BOUNDS, rotate by a fixed 15 degrees to the right.
                adjust_heading(15.0f);
                moveWaypointForward(WP_TRAJECTORY, fallbackDistance);
                if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
                    navigation_state = SAFE;
                    VERBOSE_PRINT("Rotated 15°: now inside obstacle zone, switching back to SAFE\n");
                } else {
                    VERBOSE_PRINT("Rotated 15°: still out-of-bounds, remaining in OUT_OF_BOUNDS\n");
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
 