/*
 * everything_avoider_pf.c
 *
 * Updated module using a new ONNX-based model.
 * The new model resizes the image to 60x130 then crops off the left part,
 * yielding a final input tensor of shape [1, 3, 130, 40] (interpreted as [1, 130, 40, 3]
 * by the ONNX model). The model outputs a single float in [0,1] indicating a normalized
 * desired direction (0 = far left, 0.5 = straight ahead, 1 = far right).
 *
 * NOTE: Instead of converting to RGB, the NN has been retrained to accept a full three-channel YUV image.
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
 
 // Define intermediate sizes if not defined elsewhere.
 #ifndef NN_INTERMEDIATE_WIDTH
 #define NN_INTERMEDIATE_WIDTH 60
 #endif
 #ifndef NN_HEIGHT
 #define NN_HEIGHT 130
 #endif
 #ifndef NN_FINAL_WIDTH
 #define NN_FINAL_WIDTH 40
 #endif
 
 // Define IMAGE_RGB if not already defined.
 // We reuse IMAGE_RGB as the type for a 3-channel image (even though it now contains YUV).
 #ifndef IMAGE_RGB
 #define IMAGE_RGB 2
 #endif
 
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
 static int stored_type = 0;
 
 /* ---------------- Global Variables for Video Handling ---------------- */
 static pthread_mutex_t video_frame_mutex;
 static struct image_t *latest_frame = NULL;
 
 /* ---------------- Helper: Clip function ---------------- */
 static uint8_t clip(int value) {
     if (value < 0)
         return 0;
     if (value > 255)
         return 255;
     return (uint8_t)value;
 }
 
 /* ---------------- Helper: Convert YUV422 (UYVY) to YUV ----------------
  * Converts the downsampled UYVY image to a full YUV image with 3 channels per pixel.
  * Each pixel in the output gets:
  *     - Y: from the corresponding Y value in UYVY.
  *     - U and V: taken from the shared U and V for each pair.
  */
 static void convert_yuv422_to_yuv(const struct image_t *yuv_in, struct image_t *yuv_out) {
     int width = yuv_in->w;
     int height = yuv_in->h;
     uint8_t *yuv_buf = (uint8_t*) yuv_in->buf;
     uint8_t *out_buf = (uint8_t*) yuv_out->buf;
 
     VERBOSE_PRINT("Converting YUV422 to YUV for image %dx%d\n", width, height);
     // UYVY: 4 bytes for every 2 pixels.
     for (int row = 0; row < height; row++) {
         int row_start_in = row * width * 2;  // 2 bytes per pixel
         int row_start_out = row * width * 3;   // 3 bytes per pixel in output
         for (int i = 0; i < width / 2; i++) {
             int base = row_start_in + i * 4;
             uint8_t U = yuv_buf[base];
             uint8_t Y0 = yuv_buf[base + 1];
             uint8_t V = yuv_buf[base + 2];
             uint8_t Y1 = yuv_buf[base + 3];
 
             // First pixel: channels = [Y0, U, V]
             int out_index = row_start_out + (2 * i) * 3;
             out_buf[out_index]     = Y0;
             out_buf[out_index + 1] = U;
             out_buf[out_index + 2] = V;
 
             // Second pixel: channels = [Y1, U, V]
             out_index = row_start_out + (2 * i + 1) * 3;
             out_buf[out_index]     = Y1;
             out_buf[out_index + 1] = U;
             out_buf[out_index + 2] = V;
         }
     }
 }
 
 static uint64_t last_frame_checksum = 0;

 static struct image_t *video_callback(struct image_t *img, unsigned char id) {
     (void)id;
     pthread_mutex_lock(&video_frame_mutex);
 
     // compute checksum
     uint64_t checksum = 0;
     uint8_t *buf_ptr = (uint8_t *)img->buf;
     for (int i = 0; i < img->buf_size; i += 64) {
        checksum += buf_ptr[i];
     }
     VERBOSE_PRINT("Video frame checksum: %llu (prev: %llu)\n", checksum, last_frame_checksum);
 
     latest_frame = img;
     stored_width = img->w;
     stored_height = img->h;
     stored_type = img->type;
 
     last_frame_checksum = checksum;
     pthread_mutex_unlock(&video_frame_mutex);
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
  *   1. Downsample image to an intermediate resolution (NN_INTERMEDIATE_WIDTH x NN_HEIGHT) using image_yuv422_downsample.
  *   2. Convert the downsampled UYVY image to a full three-channel YUV image (without applying any color conversion).
  *   3. Crop off the left part (20 columns) to yield a final YUV image of size NN_FINAL_WIDTH x NN_HEIGHT.
  *   4. Normalize the YUV values to [0,1] in the output tensor.
  *
  * The output tensor has shape [1, NN_HEIGHT, NN_FINAL_WIDTH, 3].
  */
/* Patch to get_camera_image_normalized() */

    static void get_camera_image_normalized(float* final_buffer) {
        int width_in = 0;
        int height_in = 0;
        uint32_t expected_buf_size = 0;

        struct image_t local_frame;
        local_frame.buf = NULL;

        // Lock and safely copy metadata and frame buffer while mutex is held
        pthread_mutex_lock(&video_frame_mutex);
        if (latest_frame && latest_frame->buf) {
            width_in = latest_frame->w;
            height_in = latest_frame->h;
            local_frame.w = latest_frame->w;
            local_frame.h = latest_frame->h;
            local_frame.type = latest_frame->type;
            expected_buf_size = width_in * height_in * 2;

            local_frame.buf_size = expected_buf_size;

            local_frame.buf = malloc(expected_buf_size);
            if (local_frame.buf != NULL) {
                memcpy(local_frame.buf, latest_frame->buf, expected_buf_size);
                VERBOSE_PRINT("Frame buffer copied inside mutex. Size: %u bytes\n", expected_buf_size);
            } else {
                VERBOSE_PRINT("Failed to malloc local_frame.buf inside mutex.\n");
            }
        } else {
            VERBOSE_PRINT("Frame or buffer was NULL inside mutex. width=%d height=%d ptr=%p\n",
                        width_in, height_in, latest_frame ? latest_frame->buf : NULL);
        }
        pthread_mutex_unlock(&video_frame_mutex);

        if (local_frame.buf == NULL) {
            VERBOSE_PRINT("Error: Frame copy failed. Aborting image normalization.\n");
            memset(final_buffer, 0, 3 * NN_HEIGHT * NN_FINAL_WIDTH * sizeof(float));
            return;
        }

        uint8_t downsample_factor = (uint8_t)(width_in / NN_INTERMEDIATE_WIDTH);
        if (downsample_factor < 1)
            downsample_factor = 1;
        VERBOSE_PRINT("Calculated downsample factor: %d\n", downsample_factor);

        struct image_t intermediate_img;
        image_create(&intermediate_img, NN_INTERMEDIATE_WIDTH, NN_HEIGHT, IMAGE_YUV422);
        if (!intermediate_img.buf) {
            VERBOSE_PRINT("Failed to allocate intermediate image.\n");
            free(local_frame.buf);
            memset(final_buffer, 0, 3 * NN_HEIGHT * NN_FINAL_WIDTH * sizeof(float));
            return;
        }

        image_yuv422_downsample(&local_frame, &intermediate_img, downsample_factor);

        struct image_t yuv_img;
        image_create(&yuv_img, NN_INTERMEDIATE_WIDTH, NN_HEIGHT, IMAGE_RGB);
        if (!yuv_img.buf) {
            VERBOSE_PRINT("Failed to allocate YUV image.\n");
            image_free(&intermediate_img);
            free(local_frame.buf);
            memset(final_buffer, 0, 3 * NN_HEIGHT * NN_FINAL_WIDTH * sizeof(float));
            return;
        }

        convert_yuv422_to_yuv(&intermediate_img, &yuv_img);

        struct image_t cropped_img;
        image_create(&cropped_img, NN_FINAL_WIDTH, NN_HEIGHT, IMAGE_RGB);
        if (!cropped_img.buf) {
            VERBOSE_PRINT("Failed to allocate cropped image.\n");
            image_free(&intermediate_img);
            image_free(&yuv_img);
            free(local_frame.buf);
            memset(final_buffer, 0, 3 * NN_HEIGHT * NN_FINAL_WIDTH * sizeof(float));
            return;
        }

        int crop_offset = NN_INTERMEDIATE_WIDTH - NN_FINAL_WIDTH;
        for (int row = 0; row < NN_HEIGHT; row++) {
            for (int col = 0; col < NN_FINAL_WIDTH; col++) {
                int src_index = (row * NN_INTERMEDIATE_WIDTH + (col + crop_offset)) * 3;
                int dst_index = (row * NN_FINAL_WIDTH + col) * 3;
                ((uint8_t*)cropped_img.buf)[dst_index]     = ((uint8_t*)yuv_img.buf)[src_index];
                ((uint8_t*)cropped_img.buf)[dst_index + 1] = ((uint8_t*)yuv_img.buf)[src_index + 1];
                ((uint8_t*)cropped_img.buf)[dst_index + 2] = ((uint8_t*)yuv_img.buf)[src_index + 2];
            }
        }

        for (int row = 0; row < NN_HEIGHT; row++) {
            for (int col = 0; col < NN_FINAL_WIDTH; col++) {
                int index = (row * NN_FINAL_WIDTH + col) * 3;
                final_buffer[index]     = ((uint8_t*)cropped_img.buf)[index] / 255.0f;
                final_buffer[index + 1] = ((uint8_t*)cropped_img.buf)[index + 1] / 255.0f;
                final_buffer[index + 2] = ((uint8_t*)cropped_img.buf)[index + 2] / 255.0f;
            }
        }

        image_free(&intermediate_img);
        image_free(&yuv_img);
        image_free(&cropped_img);
        free(local_frame.buf);
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
     int final_width = NN_FINAL_WIDTH; // 40
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
 