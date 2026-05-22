#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_FEATURE_PROVIDER_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_FEATURE_PROVIDER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"


class FeatureProvider {
 public:
  FeatureProvider(int feature_size, int8_t* feature_data);
  ~FeatureProvider();

  TfLiteStatus PopulateFeatureData(tflite::ErrorReporter* error_reporter,
                                   int32_t last_time_in_ms, int32_t time_in_ms,
                                   int* how_many_new_slices);

 private:
  int feature_size_;
  int8_t* feature_data_;
  bool is_first_run_;
};

#endif 