#include <pthread.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "modules/everything_avoider_pf/everything_avoider_pf.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "generated/flight_plan.h"
#include "state.h"
#include "modules/core/abi.h"
#include "onnxruntime_c_api.h"
#include "lib/vision/image.h"
extern "C" {
#include "modules/computer_vision/cv.h"
}
    
    

using namespace cv;

#define EVERYTHING_AVOIDER_VERBOSE TRUE
#define PRINT(string, ...) fprintf(stderr, "[everything_avoider_pf->%s()] " string, __FUNCTION__, ##__VA_ARGS__)
#if EVERYTHING_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

float maxAngleDegrees = 15.0f;
float moveDistance = 1.5f;
float fallbackDistance = 1.0f;

navigation_state_t navigation_state = SAFE;

static const OrtApi* g_ort = NULL;
static OrtEnv* g_env = NULL;
static OrtSession* g_session = NULL;
static OrtSessionOptions* g_session_options = NULL;

static pthread_mutex_t video_frame_mutex;
static struct image_t *latest_frame = NULL;
static int stored_width = 0, stored_height = 0, stored_type = 0;

static uint64_t last_frame_checksum = 0;



static struct image_t *video_callback(struct image_t *img, unsigned char id) {
    (void)id;
    pthread_mutex_lock(&video_frame_mutex);
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
    cv_add_to_device(&front_camera, video_callback, 5, 0); // 20 FPS
    VERBOSE_PRINT("Video callback registered for front camera.\n");
}

static void check_status(OrtStatus* status, const char* message) {
    if (status != nullptr) {
        const char* error_msg = g_ort->GetErrorMessage(status);
        VERBOSE_PRINT("%s: %s\n", message, error_msg);
        g_ort->ReleaseStatus(status);
        exit(1);
    }
}

static void load_depth_model(const char* model_path) {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    check_status(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "depth_model", &g_env),
                 "Failed to create ONNX environment");

    check_status(g_ort->CreateSessionOptions(&g_session_options),
                 "Failed to create session options");

    check_status(g_ort->CreateSession(g_env, model_path, g_session_options, &g_session),
                 "Failed to create ONNX session");

    VERBOSE_PRINT("ONNX session created using model at %s\n", model_path);
}

void get_camera_image_normalized(float* final_buffer) {
    int width_in = 0, height_in = 0;
    uint8_t *buf_copy = NULL;

    pthread_mutex_lock(&video_frame_mutex);
    if (latest_frame && latest_frame->buf) {
        width_in = latest_frame->w;
        height_in = latest_frame->h;
        int buf_size = width_in * height_in * 2;

        buf_copy = (uint8_t *)malloc(buf_size);
        if (buf_copy) {
            memcpy(buf_copy, latest_frame->buf, buf_size);
        }
    }
    pthread_mutex_unlock(&video_frame_mutex);

    if (!buf_copy) {
        VERBOSE_PRINT("OpenCV: Frame buffer NULL or allocation failed\n");
        memset(final_buffer, 0, 3 * NN_HEIGHT * NN_FINAL_WIDTH * sizeof(float));
        return;
    }

    Mat yuv422(height_in, width_in, CV_8UC2, buf_copy);
    Mat bgr, resized;
    cvtColor(yuv422, bgr, COLOR_YUV2BGR_Y422);
    resize(bgr, resized, Size(NN_INTERMEDIATE_WIDTH, NN_HEIGHT));

    Rect crop_region(NN_INTERMEDIATE_WIDTH - NN_FINAL_WIDTH, 0, NN_FINAL_WIDTH, NN_HEIGHT);
    Mat cropped = resized(crop_region);

    Mat float_img;
    cropped.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    int idx = 0;
    for (int row = 0; row < NN_HEIGHT; ++row) {
        for (int col = 0; col < NN_FINAL_WIDTH; ++col) {
            Vec3f pixel = float_img.at<Vec3f>(row, col);
            final_buffer[idx++] = pixel[0];
            final_buffer[idx++] = pixel[1];
            final_buffer[idx++] = pixel[2];
        }
    }

    free(buf_copy);
}

static void run_depth_inference(const float* input_tensor_data, float* output_tensor_data) {
    int64_t input_dims[4] = {1, NN_HEIGHT, NN_FINAL_WIDTH, 3};
    size_t input_tensor_size = NN_HEIGHT * NN_FINAL_WIDTH * 3;

    OrtMemoryInfo* memory_info = NULL;
    check_status(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info),
                 "Failed to create memory info");

    OrtValue* input_tensor = NULL;
    check_status(g_ort->CreateTensorWithDataAsOrtValue(memory_info, (void*)input_tensor_data,
        input_tensor_size * sizeof(float), input_dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor),
        "Failed to create tensor");

    const char* input_names[] = {"input"};
    const char* output_names[] = {"dense"};
    OrtValue* output_tensor = NULL;
    OrtStatus* status = g_ort->Run(g_session, NULL, input_names, (const OrtValue* const*)&input_tensor,
                                   1, output_names, 1, &output_tensor);
    check_status(status, "Inference failed");

    float* out;
    check_status(g_ort->GetTensorMutableData(output_tensor, (void**)&out),
                 "Failed to extract output data");

    output_tensor_data[0] = out[0];
    VERBOSE_PRINT("Inference completed. Output value: %f\n", out[0]);

    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);
}

static float process_nn_output(void) {
    size_t input_size = NN_HEIGHT * NN_FINAL_WIDTH * 3;
    float* input_buffer = (float*)malloc(input_size * sizeof(float));
    float* output_buffer = (float*)malloc(sizeof(float));

    VERBOSE_PRINT("Capturing and normalizing camera image...\n");
    get_camera_image_normalized(input_buffer);
    VERBOSE_PRINT("Image normalization complete.\n");

    VERBOSE_PRINT("Running ONNX inference...\n");
    run_depth_inference(input_buffer, output_buffer);
    float result = output_buffer[0];

    free(input_buffer);
    free(output_buffer);
    return result;
}

static void adjust_heading(float incrementDegrees) {
    float old_heading = stateGetNedToBodyEulers_f()->psi;
    float new_heading = old_heading + RadOfDeg(incrementDegrees);
    FLOAT_ANGLE_NORMALIZE(new_heading);
    nav.heading = new_heading;
    VERBOSE_PRINT("Heading adjusted from %f to %f\n", old_heading, new_heading);
}

void increase_nav_heading(float incrementDegrees) {
    adjust_heading(incrementDegrees);
}

void calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters) {
    stateCalcPositionEnu_i(); // ensure position is updated
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

void everything_avoider_pf_periodic(void) {
    if (!autopilot_in_flight()) return;

    float nn_direction = process_nn_output();
    float angle_adjustment = (nn_direction - 0.5f) * 2.0f * maxAngleDegrees;

    switch (navigation_state) {
        case SAFE:
            adjust_heading(angle_adjustment);
            moveWaypointForward(WP_TRAJECTORY, moveDistance);
            moveWaypointForward(WP_GOAL, moveDistance);
            moveWaypointForward(WP_RETREAT, -moveDistance);
            if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
                navigation_state = OUT_OF_BOUNDS;
            }
            break;
        case OUT_OF_BOUNDS:
            adjust_heading(15.0f);
            moveWaypointForward(WP_TRAJECTORY, fallbackDistance);
            if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
                navigation_state = SAFE;
            }
            break;
    }
}

void everything_avoider_pf_init(void) {
    srand((unsigned)time(NULL));
    load_depth_model("sw/airborne/modules/everything_avoider_pf/depth_cnn_model_epoch_rgb.onnx");
    pthread_mutex_init(&video_frame_mutex, NULL);
    init_video_callback();
    navigation_state = SAFE;
}
