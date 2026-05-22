#include "detection_responder.h"

extern void update_detection_results(
float person_score,
float no_person_score);

void RespondToDetection(
tflite::ErrorReporter* error_reporter,
float person_score,
float no_person_score) {

int person_score_int =
person_score * 100;

int no_person_score_int =
no_person_score * 100;

update_detection_results(
person_score_int,
no_person_score_int);

TF_LITE_REPORT_ERROR(
error_reporter,
"Person:%d%% NoPerson:%d%%",
person_score_int,
no_person_score_int);

}