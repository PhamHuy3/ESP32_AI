#include <TensorFlowLite_ESP32.h>
#include <WiFi.h>
#include <Preferences.h>
#include <WiFiClientSecure.h>
#include <ArduinoJson.h>
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
#include <lwip/sockets.h>

const char* wifi_ssid = "";
const char* wifi_password = "";

const char* TELEGRAM_BOT_TOKEN = "8131600374:AAF..........VOljyy7QHqvASPcHA";// Từ BotFather
const char* TELEGRAM_CHAT_ID = "567....75";// Từ @userinfobot
#define LED_GPIO 38

// TCP server
const int TCP_PORT = 8888;
WiFiServer tcpServer(TCP_PORT);
WiFiClient micClient;
bool micConnected = false;
unsigned long lastHeartbeat = 0;
unsigned long lastDataReceived = 0;

// camera
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

// web stream
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESP32-S3 AI CAMERA</title>
<style>
body{background:#111;color:white;font-family:Arial;text-align:center;margin:0;padding:0;}
h1{background:#222;padding:15px;margin:0;}
img{width:95%;max-width:800px;border-radius:10px;margin-top:20px;border:3px solid #444;}
.card{width:95%;max-width:800px;margin:auto;margin-top:20px;background:#222;padding:20px;border-radius:10px;}
.value{font-size:28px;font-weight:bold;}
.detect{color:#ff4444;}
.none{color:#44ff44;}
.status{margin:10px;padding:10px;border-radius:10px;}
.connected{background:#004400;color:#00ff00;}
.disconnected{background:#440000;color:#ff0000;}
</style>
</head>
<body>
<h1>ESP32-S3 PERSON DETECTION</h1>
<img src="/stream">
<div class="card">
<div id="status" class="value">Loading...</div>
<div>Person Score: <span id="person">0</span>%</div>
<div>No Person Score: <span id="noperson">0</span>%</div>
<div id="micStatus" class="status disconnected">MIC: DISCONNECTED</div>
</div>
<script>
var source = new EventSource('/events');
source.onmessage = function(event){
let data = JSON.parse(event.data);
document.getElementById("person").innerHTML = data.person_score;
document.getElementById("noperson").innerHTML = data.no_person_score;
let status = document.getElementById("status");
if(data.detected){
status.innerHTML = "PERSON DETECTED";
status.className = "value detect";
}else{
status.innerHTML = "NO PERSON";
status.className = "value none";
}
if(data.mic_connected){
document.getElementById("micStatus").innerHTML = "MIC: CONNECTED";
document.getElementById("micStatus").className = "status connected";
}else{
document.getElementById("micStatus").innerHTML = "MIC: DISCONNECTED";
document.getElementById("micStatus").className = "status disconnected";
}
}
</script>
</body>
</html>
)rawliteral";

// global
Preferences preferences;
SemaphoreHandle_t camera_mutex = NULL;
bool camera_initialized = false;
static httpd_handle_t camera_httpd = NULL;

static float current_person_score = 0;
static float current_no_person_score = 0;
static bool current_detection = false;
static bool current_mic_connected = false;

unsigned long lastPhotoTime = 0;
unsigned long lastPersonDetectedTime = 0;
bool lastDetectionState = false;

// tflite
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

constexpr int kTensorArenaSize = 81 * 1024 + scratchBufSize;
static uint8_t* tensor_arena;
}

void initCamera();
void initWiFi();
void startCameraServer();
void update_detection_results(float person_score, float no_person_score);
void sendToMic(String command);
String receiveFromMic();
void checkMicConnection();
void sendPhotoToTelegramWithButtons(camera_fb_t* fb);
void sendTelegramMessage(String message);
void checkTelegramCallback();
String urlEncode(String str);
void handleTelegramApproval(String action);

void initCamera() {
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
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return;
    }

    sensor_t* s = esp_camera_sensor_get();
    s->set_framesize(s, FRAMESIZE_QVGA);
    s->set_vflip(s, 1);
    camera_initialized = true;
    Serial.println("Camera initialized");
}

void initWiFi() {
    WiFi.begin(wifi_ssid, wifi_password);
    Serial.print("Connecting to WiFi");
    int retry = 0;
    while (WiFi.status() != WL_CONNECTED && retry < 30) {
        delay(500);
        Serial.print(".");
        retry++;
    }
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi Connected");
        Serial.print("IP: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\nWiFi Failed - Check credentials!");
    }
}

void initTCPServer() {
    tcpServer.begin();
    tcpServer.setNoDelay(true);
    Serial.print("TCP Server started on port: ");
    Serial.println(TCP_PORT);
}

void checkMicConnection() {
    if (!micConnected) {
        if (tcpServer.hasClient()) {
            if (micClient.connected()) {
                micClient.stop();
            }
            micClient = tcpServer.available();
            
            int keepAlive = 1;
            int keepIdle = 5;
            int keepInterval = 3;
            int keepCount = 3;
            micClient.setSocketOption(SOL_SOCKET, SO_KEEPALIVE, &keepAlive, sizeof(keepAlive));
            micClient.setSocketOption(IPPROTO_TCP, TCP_KEEPIDLE, &keepIdle, sizeof(keepIdle));
            micClient.setSocketOption(IPPROTO_TCP, TCP_KEEPINTVL, &keepInterval, sizeof(keepInterval));
            micClient.setSocketOption(IPPROTO_TCP, TCP_KEEPCNT, &keepCount, sizeof(keepCount));
            
            micConnected = true;
            current_mic_connected = true;
            lastDataReceived = millis();
            Serial.println("MIC connected via WiFi!");
            sendToMic("INIT");
            delay(100);
        }
    } else {
        if (millis() - lastDataReceived > 25000) {
            Serial.println("MIC timeout - disconnecting!");
            micConnected = false;
            current_mic_connected = false;
            micClient.stop();
        }
        
        if (micConnected && (millis() - lastHeartbeat > 15000)) {
            sendToMic("PING");
            lastHeartbeat = millis();
        }
    }
}

void sendToMic(String command) {
    if (micConnected && micClient.connected()) {
        micClient.println(command);
        micClient.flush();
        Serial.print("Sent to MIC: ");
        Serial.println(command);
    }
}

String receiveFromMic() {
    if (micConnected && micClient.available()) {
        String response = micClient.readStringUntil('\n');
        response.trim();
        if (response.length() > 0) {
            lastDataReceived = millis();
            Serial.print("Received from MIC: ");
            Serial.println(response);
        }
        return response;
    }
    return "";
}

// telegram 
void sendPhotoToTelegramWithButtons(camera_fb_t* fb) {
    if (String(TELEGRAM_BOT_TOKEN).length() < 20) {
        Serial.println("Telegram Bot Token not configured!");
        return;
    }
    
    WiFiClientSecure client;
    client.setInsecure();
    
    String inlineKeyboard = "{\"inline_keyboard\":[["
                            "{\"text\":\"ACCESS\",\"callback_data\":\"access\"},"
                            "{\"text\":\"DENIE\",\"callback_data\":\"denie\"}"
                            "]]}";
    
    String caption = "Khong nhan dien duoc nguoi.\nVui long xac nhan:";
    
    String boundary = "----WebKitFormBoundary" + String(millis());
    String bodyStart = "--" + boundary + "\r\n" +
                       "Content-Disposition: form-data; name=\"photo\"; filename=\"detection.jpg\"\r\n" +
                       "Content-Type: image/jpeg\r\n\r\n";
    String bodyMiddle = "\r\n--" + boundary + "\r\n" +
                        "Content-Disposition: form-data; name=\"caption\"\r\n\r\n" +
                        caption + "\r\n";
    String bodyEnd = "--" + boundary + "\r\n" +
                     "Content-Disposition: form-data; name=\"reply_markup\"\r\n\r\n" +
                     inlineKeyboard + "\r\n--" + boundary + "--\r\n";
    
    int totalLength = bodyStart.length() + fb->len + bodyMiddle.length() + bodyEnd.length();
    
    if (!client.connect("api.telegram.org", 443)) {
        Serial.println("Telegram connection failed!");
        return;
    }
    
    String url = "https://api.telegram.org/bot" + String(TELEGRAM_BOT_TOKEN) + "/sendPhoto?chat_id=" + TELEGRAM_CHAT_ID;
    
    client.print("POST " + url + " HTTP/1.1\r\n");
    client.print("Host: api.telegram.org\r\n");
    client.print("Content-Length: " + String(totalLength) + "\r\n");
    client.print("Content-Type: multipart/form-data; boundary=" + boundary + "\r\n\r\n");
    client.print(bodyStart);
    
    for (size_t i = 0; i < fb->len; i += 1024) {
        size_t chunk = (fb->len - i < 1024) ? (fb->len - i) : 1024;
        client.write(fb->buf + i, chunk);
        delay(1);
    }
    
    client.print(bodyMiddle);
    client.print(bodyEnd);
    
    while (client.available()) client.readString();
    client.stop();
    Serial.println("Photo sent to Telegram with buttons!");
}

void sendTelegramMessage(String message) {
    if (String(TELEGRAM_BOT_TOKEN).length() < 20) return;
    
    WiFiClientSecure client;
    client.setInsecure();
    
    String url = "https://api.telegram.org/bot" + String(TELEGRAM_BOT_TOKEN) + 
                 "/sendMessage?chat_id=" + TELEGRAM_CHAT_ID + 
                 "&text=" + urlEncode(message);
    
    if (client.connect("api.telegram.org", 443)) {
        client.print("GET " + url + " HTTP/1.1\r\n");
        client.print("Host: api.telegram.org\r\n");
        client.print("Connection: close\r\n\r\n");
        Serial.println("Telegram message sent");
    }
    client.stop();
}

void checkTelegramCallback() {
    if (String(TELEGRAM_BOT_TOKEN).length() < 20) return;
    
    WiFiClientSecure client;
    client.setInsecure();
    
    String url = "https://api.telegram.org/bot" + String(TELEGRAM_BOT_TOKEN) + "/getUpdates?timeout=2&offset=-1";
    
    if (!client.connect("api.telegram.org", 443)) return;
    
    client.print("GET " + url + " HTTP/1.1\r\n");
    client.print("Host: api.telegram.org\r\n");
    client.print("Connection: close\r\n\r\n");
    
    String response = "";
    unsigned long timeout = millis() + 5000;
    while (millis() < timeout && !client.available()) delay(10);
    while (client.available()) response += client.readString();
    client.stop();
    
    if (response.indexOf("\"data\":\"access\"") > 0) {
        handleTelegramApproval("access");
    } else if (response.indexOf("\"data\":\"denie\"") > 0) {
        handleTelegramApproval("denie");
    }
}

void handleTelegramApproval(String action) {
    if (action == "access") {
        Serial.println("ACCESS granted via Telegram!");
        sendToMic("ENABLE_VOICE");
        sendTelegramMessage("DA DUYET! Ban co 10 giay de noi ON hoac OFF.");
    } else if (action == "denie") {
        Serial.println("DENIE via Telegram!");
        sendToMic("DISABLE_VOICE");
        sendTelegramMessage("DA TU CHOI! Khong the dieu khien den.");
    }
}

String urlEncode(String str) {
    String encoded = "";
    for (int i = 0; i < str.length(); i++) {
        char c = str.charAt(i);
        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            encoded += c;
        } else if (c == ' ') {
            encoded += "%20";
        } else {
            char hex[4];
            sprintf(hex, "%%%02X", (unsigned char)c);
            encoded += hex;
        }
    }
    return encoded;
}

// web server
void update_detection_results(float person_score, float no_person_score) {
    current_person_score = person_score;
    current_no_person_score = no_person_score;
    current_detection = (person_score > no_person_score);
    
    if (current_detection && !lastDetectionState) {
        Serial.println("PERSON DETECTED! Enabling voice control for 10 seconds...");
        sendToMic("ENABLE_VOICE");
        lastPersonDetectedTime = millis();
    }
    lastDetectionState = current_detection;
}

static esp_err_t index_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, index_html, strlen(index_html));
}

static esp_err_t events_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/event-stream");
    while (true) {
        char buffer[256];
        sprintf(buffer,
            "data: {\"person_score\":%.0f,"
            "\"no_person_score\":%.0f,"
            "\"detected\":%s,"
            "\"mic_connected\":%s}\n\n",
            current_person_score,
            current_no_person_score,
            current_detection ? "true" : "false",
            current_mic_connected ? "true" : "false");
        
        if (httpd_resp_send_chunk(req, buffer, strlen(buffer)) != ESP_OK) break;
        vTaskDelay(pdMS_TO_TICKS(200));
    }
    return ESP_OK;
}

static esp_err_t stream_handler(httpd_req_t *req) {
    camera_fb_t * fb = NULL;
    esp_err_t res = ESP_OK;
    size_t _jpg_buf_len = 0;
    uint8_t * _jpg_buf = NULL;
    char part_buf[64];
    
    res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;
    
    while (true) {
        if (xSemaphoreTake(camera_mutex, portMAX_DELAY) == pdTRUE) {
            fb = esp_camera_fb_get();
            xSemaphoreGive(camera_mutex);
        }
        if (!fb) {
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
        }
        
        if (fb->format != PIXFORMAT_JPEG) {
            bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
            esp_camera_fb_return(fb);
            fb = NULL;
            if (!jpeg_converted) continue;
        } else {
            _jpg_buf_len = fb->len;
            _jpg_buf = fb->buf;
        }
        
        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        if (res == ESP_OK) {
            size_t hlen = snprintf(part_buf, 64, _STREAM_PART, _jpg_buf_len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        }
        if (res == ESP_OK) {
            res = httpd_resp_send_chunk(req, (const char*)_jpg_buf, _jpg_buf_len);
        }
        
        if (fb) {
            esp_camera_fb_return(fb);
            fb = NULL;
        } else if (_jpg_buf) {
            free(_jpg_buf);
            _jpg_buf = NULL;
        }
        
        if (res != ESP_OK) break;
        vTaskDelay(pdMS_TO_TICKS(30));
    }
    return res;
}

void startCameraServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;
    
    httpd_uri_t index_uri = {"/", HTTP_GET, index_handler, NULL};
    httpd_uri_t stream_uri = {"/stream", HTTP_GET, stream_handler, NULL};
    httpd_uri_t events_uri = {"/events", HTTP_GET, events_handler, NULL};
    
    if (httpd_start(&camera_httpd, &config) == ESP_OK) {
        httpd_register_uri_handler(camera_httpd, &index_uri);
        httpd_register_uri_handler(camera_httpd, &stream_uri);
        httpd_register_uri_handler(camera_httpd, &events_uri);
        Serial.println("Web server started on port 80");
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    pinMode(LED_GPIO, OUTPUT);
    digitalWrite(LED_GPIO, LOW);

    camera_mutex = xSemaphoreCreateMutex();
    
    initCamera();
    initWiFi();
    initTCPServer();
    
    if (WiFi.status() == WL_CONNECTED) {
        startCameraServer();
        Serial.print("Open browser: http://");
        Serial.println(WiFi.localIP());
        sendTelegramMessage("ESP32-CAM da khoi dong!\n IP: " + WiFi.localIP().toString());
    }
    
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    
    model = tflite::GetModel(g_person_detect_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch");
        return;
    }
    
    tensor_arena = (uint8_t*) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    
    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    micro_op_resolver.AddAveragePool2D();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();
    
    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors failed");
        return;
    }
    
    input = interpreter->input(0);
    
    if (InitCamera(error_reporter) != kTfLiteOk) {
        Serial.println("InitCamera failed");
        return;
    }
    
    Serial.println("System Ready!");
}

void loop() {
    checkMicConnection();
    checkTelegramCallback();
    
    String micResponse = receiveFromMic();
    if (micResponse == "LED_ON") {
    Serial.println("MIC confirmed LED ON");
    digitalWrite(LED_GPIO, HIGH);
} else if (micResponse == "LED_OFF") {
    Serial.println("MIC confirmed LED OFF");
    digitalWrite(LED_GPIO, LOW);
}
    
    if (GetImage(error_reporter, kNumCols, kNumRows, kNumChannels, input->data.int8) != kTfLiteOk) {
        delay(100);
        return;
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
        delay(100);
        return;
    }
    
    TfLiteTensor* output = interpreter->output(0);
    int8_t person_score = output->data.uint8[kPersonIndex];
    int8_t no_person_score = output->data.uint8[kNotAPersonIndex];
    
    float person_score_f = (person_score - output->params.zero_point) * output->params.scale;
    float no_person_score_f = (no_person_score - output->params.zero_point) * output->params.scale;
    
    update_detection_results(person_score_f * 100, no_person_score_f * 100);
    RespondToDetection(error_reporter, person_score_f, no_person_score_f);
    
    static unsigned long lastPhotoTime = 0;
    if (person_score_f < 0.3 && (millis() - lastPhotoTime) > 10000 && WiFi.status() == WL_CONNECTED) {
        lastPhotoTime = millis();
        Serial.println("No person detected! Taking photo...");
        
        camera_fb_t* fb = esp_camera_fb_get();
        if (fb) {
            sendPhotoToTelegramWithButtons(fb);
            esp_camera_fb_return(fb);
        }
    }
    
    delay(100);
}
