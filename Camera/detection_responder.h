#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_DETECTION_RESPONDER_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_PERSON_DETECTION_DETECTION_RESPONDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

void RespondToDetection(tflite::ErrorReporter* error_reporter,
                        float person_score, float no_person_score);

#endif