/*
 * video_debug.c
 *
 * This module registers a video callback to log key information about incoming video frames.
 * It prints the frame dimensions, the first 16 bytes (in hex) of the frame buffer,
 * and calculates the average Y value (assuming a UYVY format, where each 4-byte group is [U, Y, V, Y]).
 *
 * Use this module for debugging the video format.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <stdint.h>
 #include <string.h>
 #include "lib/vision/image.h"
 #include "modules/computer_vision/cv.h"  // Provides cv_add_to_device()
 #include "firmwares/rotorcraft/navigation.h"  // for front_camera definition
 
 // Verbose debug macro
 #define VIDEO_DEBUG_VERBOSE 1
 #if VIDEO_DEBUG_VERBOSE
   #define DEBUG_PRINT(fmt, ...) fprintf(stderr, "[video_debug->%s()] " fmt, __FUNCTION__, __VA_ARGS__)
 #else
   #define DEBUG_PRINT(fmt, ...) 
 #endif
 
 /*
  * video_debug_callback
  *
  * This callback is called every time a new video frame is available.
  * It logs:
  *   - Frame dimensions.
  *   - The first 16 bytes of the frame buffer (interpreted as UYVY).
  *   - The average luminance value (Y) computed from the UYVY data.
  */
 struct image_t * video_debug_callback(struct image_t *img, unsigned char id) {
     (void)id;  // Unused
     if (img == NULL) {
         DEBUG_PRINT("Received NULL image\n", "");
         return img;
     }
     DEBUG_PRINT("Frame received with dimensions: %d x %d\n", img->w, img->h);
     if (img->buf == NULL) {
         DEBUG_PRINT("Frame buffer is NULL\n", "");
         return img;
     }
     // Cast the buffer to a uint8_t pointer so we can index it
     uint8_t *buf = (uint8_t *)img->buf;
     
     // Print the first 16 bytes of the buffer
     char hex_str[128] = {0};
     int len = 16;
     // Typically, for UYVY, the buffer size should be (width * height * 2) bytes.
     // Adjust len if necessary.
     if (len > (img->w * img->h * 2)) {
          len = img->w * img->h * 2;
     }
     for (int i = 0; i < len; i++) {
          char temp[4];
          sprintf(temp, "%02x ", buf[i]);
          strcat(hex_str, temp);
     }
     DEBUG_PRINT("First 16 bytes of frame->buf: %s\n", hex_str);
     
     // If the format is UYVY, compute average luminance (Y values occur at byte indices 1 and 3 in each 4-byte group)
     int total_pixels = img->w * img->h;
     int num_groups = total_pixels / 2;
     if (num_groups > 0) {
          unsigned long long sum_y = 0;
          for (int i = 0; i < num_groups; i++) {
              int base = i * 4;
              sum_y += buf[base + 1];
              sum_y += buf[base + 3];
          }
          double avg_y = sum_y / (double)(num_groups * 2);
          DEBUG_PRINT("Average Y value: %f\n", avg_y);
     }
     
     // Return the image unmodified so it can be passed to the next module.
     return img;
 }
 
 /*
  * video_debug_init
  *
  * Registers the video_debug_callback with the front camera.
  * You can adjust the FPS (here set to 5 fps) for debugging purposes.
  */
 void video_debug_init(void) {
     cv_add_to_device(&front_camera, video_debug_callback, 5, 0);
     DEBUG_PRINT("Video debug callback registered for front camera.\n", "");
 }
 