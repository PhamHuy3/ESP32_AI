#include "command_responder.h"

void RespondToCommand(tflite::ErrorReporter* error_reporter,
                      int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
  if (is_new_command) {
    TF_LITE_REPORT_ERROR(error_reporter, "Heard %s (%d%%) @%dms", found_command,
                         score, current_time);
  }
}