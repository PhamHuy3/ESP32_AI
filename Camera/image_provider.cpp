#include <cstdlib>
#include <cstring>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "image_provider.h"
#include "model_settings.h"

static const char* TAG =
"image_provider";

extern SemaphoreHandle_t
camera_mutex;

static bool camera_ready =
false;

TfLiteStatus InitCamera(
tflite::ErrorReporter* error_reporter){

camera_ready = true;

ESP_LOGI(TAG,
"Camera ready for inference");

return kTfLiteOk;

}

TfLiteStatus GetImage(
    tflite::ErrorReporter* error_reporter,
    int image_width,
    int image_height,
    int channels,
    int8_t* image_data){

    if(!camera_ready){
        ESP_LOGE(TAG, "Camera not ready");
        return kTfLiteError;
    }

    camera_fb_t* fb = NULL;
    
    if (xSemaphoreTake(camera_mutex, portMAX_DELAY) == pdTRUE) {
        fb = esp_camera_fb_get();
        xSemaphoreGive(camera_mutex);
    }
    
    if (!fb) {
        ESP_LOGE(TAG, "Camera capture failed");
        return kTfLiteError;
    }

    bool success = false;
    uint8_t* grayscale_data = NULL;

    if(fb->format == PIXFORMAT_JPEG){
        uint8_t* rgb_data = (uint8_t*)malloc(image_width * image_height * 3);
        if(rgb_data){
            // Convert JPEG to RGB888
            if(fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, rgb_data)){
                // Convert RGB888 to grayscale
                grayscale_data = (uint8_t*)malloc(image_width * image_height);
                if(grayscale_data){
                    for(int i = 0; i < image_width * image_height; i++){
                        int r = rgb_data[i*3];
                        int g = rgb_data[i*3+1];
                        int b = rgb_data[i*3+2];
                        grayscale_data[i] = (r * 77 + g * 150 + b * 29) >> 8;
                    }
                    success = true;
                }
            }
            free(rgb_data);
        }
    }
    else if(fb->format == PIXFORMAT_GRAYSCALE){
        grayscale_data = fb->buf;
        success = true;
    }
    else if(fb->format == PIXFORMAT_RGB565){
        grayscale_data = (uint8_t*)malloc(image_width * image_height);
        if(grayscale_data){
            uint16_t* rgb565 = (uint16_t*)fb->buf;
            for(int i = 0; i < image_width * image_height; i++){
                uint16_t pixel = rgb565[i];
                int r = (pixel >> 11) & 0x1F;
                int g = (pixel >> 5) & 0x3F;
                int b = pixel & 0x1F;
                r = (r * 255) / 31;
                g = (g * 255) / 63;
                b = (b * 255) / 31;
                grayscale_data[i] = (r * 77 + g * 150 + b * 29) >> 8;
            }
            success = true;
        }
    }

    if(success && grayscale_data){
        for(int i = 0; i < image_width * image_height; i++){
            image_data[i] = grayscale_data[i] ^ 0x80;
        }
        
        if(grayscale_data != fb->buf){
            free(grayscale_data);
        }
    } else {
        ESP_LOGE(TAG, "Image conversion failed");
    }

    esp_camera_fb_return(fb);
    
    return success ? kTfLiteOk : kTfLiteError;
}