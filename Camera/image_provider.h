#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_IMAGE_PROVIDER_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_IMAGE_PROVIDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#ifndef CONFIG_PERSON_DETECTION_STATIC

void *image_provider_get_display_buf();

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
                      int image_height, int channels, int8_t* image_data);

TfLiteStatus InitCamera(tflite::ErrorReporter* error_reporter);

#endif

#endif