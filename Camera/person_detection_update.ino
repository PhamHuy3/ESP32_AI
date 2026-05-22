#include <TensorFlowLite_ESP32.h>
#include <WiFi.h>
#include <Preferences.h>
#include "esp_camera.h"
#include "esp_http_server.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>

//  camera pin
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM     4
#define SIOC_GPIO_NUM     5

#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM       8
#define Y3_GPIO_NUM       9
#define Y2_GPIO_NUM       11

#define VSYNC_GPIO_NUM    6
#define HREF_GPIO_NUM     7
#define PCLK_GPIO_NUM     13

//  web stream
#define PART_BOUNDARY "123456789000000000000987654321"

static const char* _STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;

static const char* _STREAM_BOUNDARY =
    "\r\n--" PART_BOUNDARY "\r\n";

static const char* _STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESP32 AI CAMERA</title>

<style>

body{
background:#111;
color:white;
font-family:Arial;
text-align:center;
margin:0;
padding:0;
}

h1{
background:#222;
padding:15px;
margin:0;
}

img{
width:95%;
max-width:800px;
border-radius:10px;
margin-top:20px;
border:3px solid #444;
}

.card{
width:95%;
max-width:800px;
margin:auto;
margin-top:20px;
background:#222;
padding:20px;
border-radius:10px;
}

.value{
font-size:28px;
font-weight:bold;
}

.detect{
color:#ff4444;
}

.none{
color:#44ff44;
}

</style>
</head>

<body>

<h1>ESP32-S3 PERSON DETECTION</h1>

<img src="/stream">

<div class="card">

<div id="status" class="value">Loading...</div>

<br>

<div>
Person Score:
<span id="person">0</span>
%
</div>

<br>

<div>
No Person Score:
<span id="noperson">0</span>
%
</div>

</div>

<script>

var source = new EventSource('/events');

source.onmessage = function(event){

let data = JSON.parse(event.data);

document.getElementById("person").innerHTML =
data.person_score;

document.getElementById("noperson").innerHTML =
data.no_person_score;

let status = document.getElementById("status");

if(data.detected){

status.innerHTML = "PERSON DETECTED";
status.className = "value detect";

}
else{

status.innerHTML = "NO PERSON";
status.className = "value none";

}

}

</script>

</body>
</html>
)rawliteral";

//  global
Preferences preferences;

SemaphoreHandle_t camera_mutex = NULL;

bool camera_initialized = false;

static httpd_handle_t camera_httpd = NULL;

static float current_person_score = 0;
static float current_no_person_score = 0;
static bool current_detection = false;

//  tflite
namespace {

tflite::ErrorReporter* error_reporter = nullptr;

const tflite::Model* model = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;

#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 39 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif

constexpr int kTensorArenaSize =
81 * 1024 + scratchBufSize;

static uint8_t* tensor_arena;

}

void initCamera();
void initWiFi();
void connectToWiFi(const char* ssid,const char* pass);
void startCameraServer();

void update_detection_results(float person_score,
                              float no_person_score);

void initCamera(){

camera_config_t config;

config.ledc_channel = LEDC_CHANNEL_0;
config.ledc_timer = LEDC_TIMER_0;

config.pin_d0 = Y2_GPIO_NUM;
config.pin_d1 = Y3_GPIO_NUM;
config.pin_d2 = Y4_GPIO_NUM;
config.pin_d3 = Y5_GPIO_NUM;
config.pin_d4 = Y6_GPIO_NUM;
config.pin_d5 = Y7_GPIO_NUM;
config.pin_d6 = Y8_GPIO_NUM;
config.pin_d7 = Y9_GPIO_NUM;

config.pin_xclk = XCLK_GPIO_NUM;
config.pin_pclk = PCLK_GPIO_NUM;
config.pin_vsync = VSYNC_GPIO_NUM;
config.pin_href = HREF_GPIO_NUM;

config.pin_sccb_sda = SIOD_GPIO_NUM;
config.pin_sccb_scl = SIOC_GPIO_NUM;

config.pin_pwdn = PWDN_GPIO_NUM;
config.pin_reset = RESET_GPIO_NUM;

config.xclk_freq_hz = 20000000;

config.pixel_format = PIXFORMAT_JPEG;

config.frame_size = FRAMESIZE_QVGA;

config.jpeg_quality = 12;

config.fb_count = 2;

config.fb_location = CAMERA_FB_IN_PSRAM;

config.grab_mode = CAMERA_GRAB_LATEST;

esp_err_t err = esp_camera_init(&config);

if(err != ESP_OK){

Serial.printf("Camera init failed: 0x%x\n",err);
return;

}

sensor_t* s = esp_camera_sensor_get();

s->set_framesize(s,FRAMESIZE_QVGA);
s->set_vflip(s,1);

camera_initialized = true;

Serial.println("Camera init OK");

}

void connectToWiFi(const char* ssid,const char* pass){

WiFi.begin(ssid,pass);

Serial.print("Connecting");

int retry = 0;

while(WiFi.status()!=WL_CONNECTED && retry<20){

delay(500);
Serial.print(".");
retry++;

}

if(WiFi.status()==WL_CONNECTED){

Serial.println("");
Serial.println("WiFi Connected");
Serial.println(WiFi.localIP());

}else{

Serial.println("");
Serial.println("WiFi Failed");

}

}

void initWiFi(){

preferences.begin("wifi",false);

String ssid =
preferences.getString("ssid","");

String pass =
preferences.getString("password","");

if(ssid.length()>0){

connectToWiFi(ssid.c_str(),pass.c_str());

if(WiFi.status()==WL_CONNECTED){

preferences.end();
return;

}

}

while(Serial.available()) Serial.read();

Serial.println("Enter SSID:");

while(Serial.available()==0){

delay(10);

}

String newSSID =
Serial.readStringUntil('\n');

newSSID.trim();

Serial.println("Enter PASSWORD:");

while(Serial.available()==0){

delay(10);

}

String newPASS =
Serial.readStringUntil('\n');

newPASS.trim();

connectToWiFi(newSSID.c_str(),
              newPASS.c_str());

if(WiFi.status()==WL_CONNECTED){

preferences.putString("ssid",newSSID);
preferences.putString("password",newPASS);

Serial.println("WiFi Saved");

}

preferences.end();

}

void update_detection_results(float person_score,
                              float no_person_score){

current_person_score = person_score;
current_no_person_score = no_person_score;

current_detection =
(person_score > no_person_score);

}

static esp_err_t index_handler(httpd_req_t *req){

httpd_resp_set_type(req,"text/html");

return httpd_resp_send(req,
                       index_html,
                       strlen(index_html));

}

static esp_err_t events_handler(httpd_req_t *req){

httpd_resp_set_type(req,"text/event-stream");

while(true){

char buffer[128];

sprintf(buffer,
"data: {\"person_score\":%.0f,"
"\"no_person_score\":%.0f,"
"\"detected\":%s}\n\n",

current_person_score,
current_no_person_score,

current_detection ?
"true":"false");

if(httpd_resp_send_chunk(req,
                         buffer,
                         strlen(buffer))!=ESP_OK){

break;

}

vTaskDelay(pdMS_TO_TICKS(200));

}

return ESP_OK;

}

//  fixed stream
static esp_err_t stream_handler(httpd_req_t *req){

camera_fb_t * fb = NULL;

esp_err_t res = ESP_OK;

size_t _jpg_buf_len = 0;

uint8_t * _jpg_buf = NULL;

char part_buf[64];

res = httpd_resp_set_type(req,
                          _STREAM_CONTENT_TYPE);

if(res != ESP_OK){

return res;

}

while(true){

if(xSemaphoreTake(camera_mutex,
                  portMAX_DELAY)==pdTRUE){

fb = esp_camera_fb_get();

xSemaphoreGive(camera_mutex);

}

if(!fb){

Serial.println("Camera capture failed");

continue;

}

if(fb->format != PIXFORMAT_JPEG){

bool jpeg_converted =
frame2jpg(fb,
          80,
          &_jpg_buf,
          &_jpg_buf_len);

esp_camera_fb_return(fb);

fb = NULL;

if(!jpeg_converted){

Serial.println("JPEG compression failed");

continue;

}

}else{

_jpg_buf_len = fb->len;
_jpg_buf = fb->buf;

}

res = httpd_resp_send_chunk(req,
                            _STREAM_BOUNDARY,
                            strlen(_STREAM_BOUNDARY));

if(res == ESP_OK){

size_t hlen =
snprintf(part_buf,
         64,
         _STREAM_PART,
         _jpg_buf_len);

res = httpd_resp_send_chunk(req,
                            part_buf,
                            hlen);

}

if(res == ESP_OK){

res = httpd_resp_send_chunk(req,
                            (const char*)_jpg_buf,
                            _jpg_buf_len);

}

if(fb){

esp_camera_fb_return(fb);

fb = NULL;

}else if(_jpg_buf){

free(_jpg_buf);
_jpg_buf = NULL;

}

if(res != ESP_OK){

break;

}

vTaskDelay(pdMS_TO_TICKS(30));

}

return res;

}

void startCameraServer(){

httpd_config_t config =
HTTPD_DEFAULT_CONFIG();

config.server_port = 80;

httpd_uri_t index_uri = {

.uri = "/",
.method = HTTP_GET,
.handler = index_handler,
.user_ctx = NULL

};

httpd_uri_t stream_uri = {

.uri = "/stream",
.method = HTTP_GET,
.handler = stream_handler,
.user_ctx = NULL

};

httpd_uri_t events_uri = {

.uri = "/events",
.method = HTTP_GET,
.handler = events_handler,
.user_ctx = NULL

};

if(httpd_start(&camera_httpd,
               &config)==ESP_OK){

httpd_register_uri_handler(camera_httpd,
                           &index_uri);

httpd_register_uri_handler(camera_httpd,
                           &stream_uri);

httpd_register_uri_handler(camera_httpd,
                           &events_uri);

Serial.println("Web server started");

}

}

void setup(){

Serial.begin(115200);

delay(1000);

camera_mutex =
xSemaphoreCreateMutex();

initCamera();

initWiFi();

if(WiFi.status()==WL_CONNECTED){

startCameraServer();

Serial.print("Open browser: http://");
Serial.println(WiFi.localIP());

}

static tflite::MicroErrorReporter
micro_error_reporter;

error_reporter =
&micro_error_reporter;

model =
tflite::GetModel(
g_person_detect_model_data);

if(model->version() !=
TFLITE_SCHEMA_VERSION){

Serial.println("Model schema mismatch");
return;

}

tensor_arena =
(uint8_t*) heap_caps_malloc(
kTensorArenaSize,
MALLOC_CAP_INTERNAL |
MALLOC_CAP_8BIT);

static tflite::MicroMutableOpResolver<5>
micro_op_resolver;

micro_op_resolver.AddAveragePool2D();
micro_op_resolver.AddConv2D();
micro_op_resolver.AddDepthwiseConv2D();
micro_op_resolver.AddReshape();
micro_op_resolver.AddSoftmax();

static tflite::MicroInterpreter
static_interpreter(

model,
micro_op_resolver,
tensor_arena,
kTensorArenaSize,
error_reporter

);

interpreter = &static_interpreter;

if(interpreter->AllocateTensors()
!= kTfLiteOk){

Serial.println("AllocateTensors failed");
return;

}

input = interpreter->input(0);

if(InitCamera(error_reporter)
!= kTfLiteOk){

Serial.println("InitCamera failed");
return;

}

Serial.println("System Ready");

}

void loop(){

if(GetImage(error_reporter,
            kNumCols,
            kNumRows,
            kNumChannels,
            input->data.int8)
!= kTfLiteOk){

Serial.println("Image capture failed");

delay(100);
return;

}

if(interpreter->Invoke()
!= kTfLiteOk){

Serial.println("Invoke failed");

delay(100);
return;

}

TfLiteTensor* output =
interpreter->output(0);

int8_t person_score =
output->data.uint8[kPersonIndex];

int8_t no_person_score =
output->data.uint8[kNotAPersonIndex];

float person_score_f =
(person_score -
output->params.zero_point)
*
output->params.scale;

float no_person_score_f =
(no_person_score -
output->params.zero_point)
*
output->params.scale;

update_detection_results(
person_score_f * 100,
no_person_score_f * 100);

RespondToDetection(
error_reporter,
person_score_f,
no_person_score_f);

delay(100);
}