#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_ESP_APP_CAMERA_ESP_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_ESP_APP_CAMERA_ESP_H_

#include "sensor.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "esp_system.h"
#include "sensor.h"
#include "esp_main.h"

#if defined DISPLAY_SUPPORT
#define CAMERA_PIXEL_FORMAT PIXFORMAT_RGB565
#else
#define CAMERA_PIXEL_FORMAT PIXFORMAT_GRAYSCALE
#endif

#define CAMERA_FRAME_SIZE FRAMESIZE_96X96
#define CAMERA_MODULE_NAME "ESP32-S3-EYE"
#define CAMERA_PIN_PWDN -1
#define CAMERA_PIN_RESET -1
#define CAMERA_PIN_XCLK 15
#define CAMERA_PIN_SIOD 4
#define CAMERA_PIN_SIOC 5

#define CAMERA_PIN_D7 16  // Y9
#define CAMERA_PIN_D6 17  // Y8
#define CAMERA_PIN_D5 18  // Y7
#define CAMERA_PIN_D4 12  // Y6
#define CAMERA_PIN_D3 10  // Y5
#define CAMERA_PIN_D2 8   // Y4
#define CAMERA_PIN_D1 9   // Y3
#define CAMERA_PIN_D0 11  // Y2
#define CAMERA_PIN_VSYNC 6
#define CAMERA_PIN_HREF 7
#define CAMERA_PIN_PCLK 13

#define XCLK_FREQ_HZ 15000000

#ifdef __cplusplus
extern "C" {
#endif

int app_camera_init();

#ifdef __cplusplus
}
#endif

#endif