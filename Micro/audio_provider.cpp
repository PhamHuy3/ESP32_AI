#include "audio_provider.h"
#include <cstdlib>
#include <cstring>

#include "freertos/FreeRTOS.h"
#include "driver/i2s.h"
#include "esp_log.h"
#include "freertos/task.h"
#include "ringbuf.h"
#include "micro_model_settings.h"

static const char* TAG = "AUDIO_PROVIDER";

#define I2S_PORT I2S_NUM_0
#define I2S_SAMPLE_RATE 16000

ringbuf_t* g_audio_capture_buffer;
volatile int32_t g_latest_audio_timestamp = 0;

constexpr int32_t history_samples_to_keep =
    ((kFeatureSliceDurationMs - kFeatureSliceStrideMs) *
     (kAudioSampleFrequency / 1000));
     
constexpr int32_t new_samples_to_get =
    (kFeatureSliceStrideMs * (kAudioSampleFrequency / 1000));

namespace {
int16_t g_audio_output_buffer[kMaxAudioSampleSize];
bool g_is_audio_initialized = false;
int16_t g_history_buffer[history_samples_to_keep];
}  // namespace

const int32_t kAudioCaptureBufferSize = 16000;
const int32_t i2s_bytes_to_read = 1600;

static void CaptureSamples(void* arg) {
  size_t bytes_read = 0;
  uint8_t* i2s_read_buffer = (uint8_t*)malloc(i2s_bytes_to_read);
  
  if (!i2s_read_buffer) {
    ESP_LOGE(TAG, "Failed to allocate I2S buffer");
    vTaskDelete(NULL);
    return;
  }
  
  while (1) {
    esp_err_t err = i2s_read(I2S_PORT, (void*)i2s_read_buffer, 
                             i2s_bytes_to_read, &bytes_read, portMAX_DELAY);
    
    if (err == ESP_OK && bytes_read > 0) {
      int bytes_written = rb_write(g_audio_capture_buffer,
                                   (uint8_t*)i2s_read_buffer, bytes_read, 10);
      
      if (bytes_written > 0) {
        g_latest_audio_timestamp += 
            ((1000 * (bytes_written / 2)) / I2S_SAMPLE_RATE);
      }
    }
    taskYIELD();
  }
  
  free(i2s_read_buffer);
  vTaskDelete(NULL);
}

TfLiteStatus InitAudioRecording(tflite::ErrorReporter* error_reporter) {
  ESP_LOGI(TAG, "Initializing audio recording...");
  
  g_audio_capture_buffer = rb_init("tf_ringbuffer", kAudioCaptureBufferSize);
  if (!g_audio_capture_buffer) {
    ESP_LOGE(TAG, "Error creating ring buffer");
    return kTfLiteError;
  }
  
  xTaskCreate(CaptureSamples, "CaptureSamples", 8192, NULL, 10, NULL);
  vTaskDelay(pdMS_TO_TICKS(100));
  
  ESP_LOGI(TAG, "Audio recording started");
  return kTfLiteOk;
}

TfLiteStatus GetAudioSamples(tflite::ErrorReporter* error_reporter,
                             int start_ms, int duration_ms,
                             int* audio_samples_size, int16_t** audio_samples) {
  if (!g_is_audio_initialized) {
    TfLiteStatus init_status = InitAudioRecording(error_reporter);
    if (init_status != kTfLiteOk) {
      return init_status;
    }
    g_is_audio_initialized = true;
  }
  
  memcpy((void*)(g_audio_output_buffer), (void*)(g_history_buffer),
         history_samples_to_keep * sizeof(int16_t));
  
  int32_t bytes_to_read = new_samples_to_get * sizeof(int16_t);
  int32_t bytes_read = rb_read(g_audio_capture_buffer,
                               ((uint8_t*)(g_audio_output_buffer + history_samples_to_keep)),
                               bytes_to_read, 100);
  
  if (bytes_read < bytes_to_read) {
    int remaining = bytes_to_read - bytes_read;
    memset((uint8_t*)(g_audio_output_buffer + history_samples_to_keep) + bytes_read, 
           0, remaining);
  }
  
  memcpy((void*)(g_history_buffer),
         (void*)(g_audio_output_buffer + new_samples_to_get),
         history_samples_to_keep * sizeof(int16_t));
  
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;
  
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { 
  return g_latest_audio_timestamp; 
}