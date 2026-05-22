#ifndef TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_
#define TENSORFLOW_LITE_MICRO_EXAMPLES_MICRO_SPEECH_MICRO_FEATURES_MICRO_MODEL_SETTINGS_H_

constexpr int kMaxAudioSampleSize = 512;
constexpr int kAudioSampleFrequency = 16000;
constexpr int kFeatureSliceSize = 40;
constexpr int kFeatureSliceCount = 49;
constexpr int kFeatureElementCount = (kFeatureSliceSize * kFeatureSliceCount);
constexpr int kFeatureSliceStrideMs = 20;
constexpr int kFeatureSliceDurationMs = 30;
constexpr int kSilenceIndex = 0;
constexpr int kUnknownIndex = 1;
constexpr int kCategoryCount = 4;
extern const char* kCategoryLabels[kCategoryCount];

#endif