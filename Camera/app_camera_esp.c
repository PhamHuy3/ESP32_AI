#include "app_camera_esp.h"

static const char *TAG = "app_camera";

int app_camera_init() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = CAMERA_PIN_D0;
  config.pin_d1 = CAMERA_PIN_D1;
  config.pin_d2 = CAMERA_PIN_D2;
  config.pin_d3 = CAMERA_PIN_D3;
  config.pin_d4 = CAMERA_PIN_D4;
  config.pin_d5 = CAMERA_PIN_D5;
  config.pin_d6 = CAMERA_PIN_D6;
  config.pin_d7 = CAMERA_PIN_D7;
  config.pin_xclk = CAMERA_PIN_XCLK;
  config.pin_pclk = CAMERA_PIN_PCLK;
  config.pin_vsync = CAMERA_PIN_VSYNC;
  config.pin_href = CAMERA_PIN_HREF;
  config.pin_sscb_sda = CAMERA_PIN_SIOD;
  config.pin_sscb_scl = CAMERA_PIN_SIOC;
  config.pin_pwdn = CAMERA_PIN_PWDN;
  config.pin_reset = CAMERA_PIN_RESET;
  config.xclk_freq_hz = XCLK_FREQ_HZ;
  config.pixel_format = CAMERA_PIXEL_FORMAT;
  config.frame_size = CAMERA_FRAME_SIZE;
  config.jpeg_quality = 10;
  config.fb_count = 2;
  
#if CONFIG_ESP32S3_PSRAM || CONFIG_ESP32_PSRAM
  config.fb_location = CAMERA_FB_IN_PSRAM;
  ESP_LOGI(TAG, "Using PSRAM for frame buffer");
#else
  config.fb_location = CAMERA_FB_IN_DRAM;
  ESP_LOGI(TAG, "Using DRAM for frame buffer");
#endif

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Camera init failed with error 0x%x", err);
    return -1;
  }
  
  sensor_t *s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_vflip(s, 1);
    s->set_hmirror(s, 0);
    s->set_brightness(s, 0);
    s->set_contrast(s, 0);
    s->set_saturation(s, 0);

    if (s->id.PID == OV3660_PID) {
      s->set_brightness(s, 1);
      s->set_saturation(s, -2);
      ESP_LOGI(TAG, "OV3660 sensor detected - applying special settings");
    } else {
      ESP_LOGI(TAG, "Sensor PID: 0x%x detected", s->id.PID);
    }
    
    ESP_LOGI(TAG, "Camera initialized successfully with %s", CAMERA_MODULE_NAME);
  } else {
    ESP_LOGW(TAG, "Could not get sensor handle");
  }
  
  return 0;
}