#include <WiFi.h>
#include <WebServer.h>
#include <driver/i2s.h>
#include <TensorFlowLite_ESP32.h>

#include "main_functions.h"
#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "model.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

// wifi config
const char* ssid = "";
const char* password = "";

// TCP config
const char* cam_ip = "";
const int cam_port = 8888;
WiFiClient tcpClient;
bool camConnected = false;
unsigned long lastHeartbeat = 0;
unsigned long lastReconnectAttempt = 0;
unsigned long lastDataReceived = 0;

// led control
#define LED_PIN 38
bool ledState = false;

// voice control
bool voiceControlEnabled = false;
unsigned long voiceControlStartTime = 0;
const unsigned long VOICE_TIMEOUT = 10000;  // 10s

// I2S pin
#define I2S_WS   42
#define I2S_SCK  41
#define I2S_SD   39
#define I2S_PORT I2S_NUM_0

WebServer server(80);

// tensorflows global
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
int32_t previous_time = 0;

constexpr int kTensorArenaSize = 35 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;

String last_command = "silence";
uint8_t last_score = 0;
unsigned long last_command_time = 0;
String history[20];
int history_index = 0;
}

void i2s_init();
void setupTensorFlow();
void processVoiceRecognition();
void connectToCAM();
void sendToCAM(String command);
String receiveFromCAM();
void processIncomingCommands();
void turnLEDOn();
void turnLEDOff();

// I2S setup
void i2s_init() {
    Serial.println("Initializing I2S...");
    const i2s_config_t config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };
    
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,
        .data_in_num = I2S_SD
    };
    
    i2s_driver_install(I2S_PORT, &config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_zero_dma_buffer(I2S_PORT);
    Serial.println("I2S initialized");
}

// TCP client
void connectToCAM() {
    if (camConnected) {
        if (!tcpClient.connected()) {
            camConnected = false;
            Serial.println("Connection lost, reconnecting...");
        }
        return;
    }
    
    if (millis() - lastReconnectAttempt < 3000) return;
    lastReconnectAttempt = millis();
    
    Serial.print("Connecting to CAM at ");
    Serial.print(cam_ip);
    Serial.print(":");
    Serial.println(cam_port);
    
    if (tcpClient.connect(cam_ip, cam_port)) {
        camConnected = true;
        lastDataReceived = millis();
        Serial.println("Connected to CAM!");
        sendToCAM("READY");
    } else {
        Serial.println("Failed to connect to CAM");
    }
}

void sendToCAM(String command) {
    if (camConnected && tcpClient.connected()) {
        tcpClient.println(command);
        tcpClient.flush();
        Serial.print("Sent to CAM: ");
        Serial.println(command);
    }
}

String receiveFromCAM() {
    if (camConnected && tcpClient.available()) {
        String response = tcpClient.readStringUntil('\n');
        response.trim();
        if (response.length() > 0) {
            lastDataReceived = millis();
            Serial.print("Received from CAM: ");
            Serial.println(response);
        }
        return response;
    }
    return "";
}

void processIncomingCommands() {
    if (!camConnected) return;
    
    if (millis() - lastDataReceived > 25000 && camConnected) {
        Serial.println("CAM timeout!");
        camConnected = false;
        tcpClient.stop();
        return;
    }
    
    String cmd = receiveFromCAM();
    if (cmd.length() > 0) {
        if (cmd == "ENABLE_VOICE") {
            Serial.println("Voice control ENABLED for 10 seconds!");
            voiceControlEnabled = true;
            voiceControlStartTime = millis();
            sendToCAM("VOICE_ENABLED");
        }
        else if (cmd == "DISABLE_VOICE") {
            Serial.println("Voice control DISABLED!");
            voiceControlEnabled = false;
            sendToCAM("VOICE_DISABLED");
        }
        else if (cmd == "PING") {
            sendToCAM("PONG");
        }
        else if (cmd == "INIT") {
            sendToCAM("READY");
            Serial.println("Handshake complete!");
        }
    }
    
    if (millis() - lastHeartbeat > 15000) {
        sendToCAM("PING");
        lastHeartbeat = millis();
    }
}

// led control
void turnLEDOn() {
    ledState = true;
    digitalWrite(LED_PIN, HIGH);
    Serial.println("LED TURNED ON");
    sendToCAM("LED_ON");
}

void turnLEDOff() {
    ledState = false;
    digitalWrite(LED_PIN, LOW);
    Serial.println("LED TURNED OFF");
    sendToCAM("LED_OFF");
}

void setupTensorFlow() {
    Serial.println("\nInitializing TensorFlow Lite...");
    
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema mismatch!");
        return;
    }
    Serial.println("Model loaded");
    
  static tflite::MicroMutableOpResolver<9> micro_op_resolver(error_reporter);
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddSoftmax();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddShape();
  micro_op_resolver.AddStridedSlice();
  micro_op_resolver.AddPack();
  micro_op_resolver.AddRelu();
    
    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors failed!");
        return;
    }
    Serial.println("Tensors allocated");
    
    model_input = interpreter->input(0);
    model_input_buffer = model_input->data.int8;
    
    static FeatureProvider static_feature_provider(kFeatureElementCount, feature_buffer);
    feature_provider = &static_feature_provider;
    
    previous_time = 0;
    Serial.println("TensorFlow Lite ready\n");
}

void handleRoot() {
    String html = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ESP32-S3 Voice Recognition</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);min-height:100vh;padding:20px;}
.container{max-width:700px;margin:0 auto;}
.header{text-align:center;color:white;margin-bottom:30px;}
.command-card{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);border-radius:30px;padding:40px;text-align:center;margin-bottom:20px;}
.command-value{font-size:64px;font-weight:bold;color:white;text-transform:uppercase;}
.score{font-size:24px;color:#00ff88;margin-top:10px;}
.confidence-bar{width:100%;height:10px;background:rgba(255,255,255,0.2);border-radius:5px;margin-top:15px;}
.confidence-fill{height:100%;background:linear-gradient(90deg,#00d4ff,#00ff88);width:0%;border-radius:5px;}
.status{background:rgba(0,0,0,0.5);border-radius:15px;padding:15px;text-align:center;color:#ffaa00;margin-bottom:20px;}
.history{background:rgba(0,0,0,0.4);border-radius:20px;padding:20px;}
.history-title{color:white;margin-bottom:15px;}
.history-item{color:#ccc;padding:8px 0;display:flex;justify-content:space-between;border-bottom:1px solid rgba(255,255,255,0.1);}
.history-command{color:#00d4ff;font-weight:bold;}
.history-score{color:#00ff88;}
button{background:linear-gradient(135deg,#667eea,#764ba2);border:none;padding:10px 25px;border-radius:25px;color:white;cursor:pointer;margin:5px;}
.led-on{background:#00ff8822;color:#00ff88;border:1px solid #00ff88;}
.led-off{background:#ff444422;color:#ff4444;border:1px solid #ff4444;}
</style>
</head>
<body>
<div class="container">
<div class="header"><h1>Voice Recognition</h1><p>ESP32-S3 | TensorFlow Lite | INMP441</p></div>
<div class="command-card">
<div class="command-value" id="commandDisplay">-</div>
<div class="score" id="scoreDisplay">0%</div>
<div class="confidence-bar"><div class="confidence-fill" id="confidenceFill"></div></div>
</div>
<div class="status" id="statusDisplay">Listening...</div>
<div id="ledStatus" class="status led-off">LED: OFF</div>
<div id="voiceStatus" class="status">Voice Control: WAITING</div>
<div class="history">
<div class="history-title">Command History</div>
<div id="historyList">No commands yet</div>
</div>
<div style="text-align:center;margin-top:20px;">
<button onclick="resetCommand()">Reset</button>
<button onclick="clearHistory()">Clear</button>
<button onclick="toggleLED()">Manual LED</button>
</div>
</div>
<script>
function refreshData(){
fetch('/command').then(r=>r.json()).then(d=>{
document.getElementById('commandDisplay').innerHTML=d.command.toUpperCase();
document.getElementById('scoreDisplay').innerHTML=d.score+'%';
document.getElementById('confidenceFill').style.width=d.score+'%';
if(d.command!=='silence'&&d.score>40){
document.getElementById('statusDisplay').innerHTML=d.command.toUpperCase();
document.getElementById('statusDisplay').style.color='#00ff88';
}else{document.getElementById('statusDisplay').innerHTML='Listening...';document.getElementById('statusDisplay').style.color='#ffaa00';}
});
fetch('/led').then(r=>r.json()).then(d=>{
let ledDiv=document.getElementById('ledStatus');
if(d.state){ledDiv.innerHTML='LED: ON';ledDiv.className='status led-on';}
else{ledDiv.innerHTML='LED: OFF';ledDiv.className='status led-off';}
});
fetch('/voice-status').then(r=>r.json()).then(d=>{
let voiceDiv=document.getElementById('voiceStatus');
if(d.enabled){voiceDiv.innerHTML='Voice Control: ENABLED ('+d.remaining+'s)';voiceDiv.style.color='#00ff88';}
else{voiceDiv.innerHTML='Voice Control: DISABLED';voiceDiv.style.color='#ff4444';}
});
}
function resetCommand(){fetch('/reset');document.getElementById('commandDisplay').innerHTML='-';document.getElementById('scoreDisplay').innerHTML='0%';document.getElementById('confidenceFill').style.width='0%';}
function clearHistory(){fetch('/clear-history');document.getElementById('historyList').innerHTML='No commands yet';}
function toggleLED(){fetch('/toggle-led');}
function loadHistory(){fetch('/history').then(r=>r.json()).then(d=>{if(d.history&&d.history.length>0){let html='';d.history.forEach(item=>{html+='<div class="history-item"><span class="history-command">'+item.command.toUpperCase()+'</span><span class="history-score">'+item.score+'%</span></div>';});document.getElementById('historyList').innerHTML=html;}});}
setInterval(refreshData,300);setInterval(loadHistory,2000);refreshData();
</script>
</body>
</html>
)rawliteral";
    server.send(200, "text/html", html);
}

void handleCommand() {
    String json = "{\"command\":\"" + last_command + "\",\"score\":" + String(last_score) + "}";
    server.send(200, "application/json", json);
}

void handleLED() {
    server.send(200, "application/json", "{\"state\":" + String(ledState ? "true" : "false") + "}");
}

void handleVoiceStatus() {
    int remaining = 0;
    if (voiceControlEnabled) {
        unsigned long elapsed = millis() - voiceControlStartTime;
        if (elapsed < VOICE_TIMEOUT) {
            remaining = (VOICE_TIMEOUT - elapsed) / 1000;
        } else {
            voiceControlEnabled = false;
        }
    }
    String json = "{\"enabled\":" + String(voiceControlEnabled ? "true" : "false") + ",\"remaining\":" + String(remaining) + "}";
    server.send(200, "application/json", json);
}

void handleToggleLED() {
    if (ledState) turnLEDOff(); else turnLEDOn();
    server.send(200, "application/json", "{\"status\":\"ok\"}");
}

void handleReset() {
    last_command = "silence";
    last_score = 0;
    server.send(200, "application/json", "{\"status\":\"ok\"}");
}

void handleClearHistory() {
    for (int i = 0; i < 20; i++) history[i] = "";
    history_index = 0;
    server.send(200, "application/json", "{\"status\":\"ok\"}");
}

void handleHistory() {
    String json = "{\"history\":[";
    bool first = true;
    for (int i = 0; i < 20; i++) {
        if (history[i] != "") {
            if (!first) json += ",";
            json += history[i];
            first = false;
        }
    }
    json += "]}";
    server.send(200, "application/json", json);
}

void processVoiceRecognition() {
    if (!feature_provider) return;
    
    const int32_t current_time = millis();
    int how_many_new_slices = 0;
    
    TfLiteStatus feature_status = feature_provider->PopulateFeatureData(error_reporter, previous_time, current_time, &how_many_new_slices);
    
    if (feature_status != kTfLiteOk || how_many_new_slices == 0) return;
    
    previous_time = current_time;
    
    for (int i = 0; i < kFeatureElementCount; i++) {
        model_input_buffer[i] = feature_buffer[i];
    }
    
    if (interpreter->Invoke() != kTfLiteOk) return;
    
    TfLiteTensor* output = interpreter->output(0);
    
    int silence_val = output->data.int8[0] + 128;
    int unknown_val = output->data.int8[1] + 128;
    int on_val = output->data.int8[2] + 128;
    int off_val = output->data.int8[3] + 128;

    
    static int debug_count = 0;
    debug_count++;
    if (debug_count % 50 == 0) {
        Serial.printf("Y:%d N:%d UNK:%d | VoiceEnabled:%d | LED:%d\n", 
                      on_val, off_val, unknown_val, voiceControlEnabled, ledState);
    }
    
    // timeout
    if (voiceControlEnabled && (millis() - voiceControlStartTime) > VOICE_TIMEOUT) {
        Serial.println("Voice control timeout! (10 seconds expired)");
        voiceControlEnabled = false;
        sendToCAM("VOICE_TIMEOUT");
    }
    
    static unsigned long last_detect = 0;
    
    if (voiceControlEnabled) {
        if (on_val > 200 && on_val > unknown_val + 30 && (current_time - last_detect) > 1500) {
            last_detect = current_time;
            Serial.printf("on! (%d)\n", on_val);
            turnLEDOn();
            voiceControlEnabled = false;
            sendToCAM("on_DETECTED");
            
            last_command = "on";
            last_score = on_val;
            String entry = "{\"command\":\"on\",\"score\":" + String(on_val) + "}";
            history[history_index % 20] = entry;
            history_index++;
            RespondToCommand(error_reporter, current_time, "on", on_val, true);
        }
        else if (off_val > 130 && off_val > unknown_val + 20 && (current_time - last_detect) > 1500) {
            last_detect = current_time;
            Serial.printf("off! (%d)\n", off_val);
            turnLEDOff();
            voiceControlEnabled = false;
            sendToCAM("off_DETECTED");
            
            last_command = "off";
            last_score = off_val;
            String entry = "{\"command\":\"off\",\"score\":" + String(off_val) + "}";
            history[history_index % 20] = entry;
            history_index++;
            RespondToCommand(error_reporter, current_time, "off", off_val, true);
        }
    }
}

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);
    
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected");
    Serial.print("  IP: ");
    Serial.println(WiFi.localIP());
    
    i2s_init();
    setupTensorFlow();
    
    server.on("/", handleRoot);
    server.on("/command", handleCommand);
    server.on("/led", handleLED);
    server.on("/voice-status", handleVoiceStatus);
    server.on("/toggle-led", handleToggleLED);
    server.on("/reset", handleReset);
    server.on("/clear-history", handleClearHistory);
    server.on("/history", handleHistory);
    server.begin();
}

void loop() {
    server.handleClient();
    connectToCAM();
    processIncomingCommands();
    processVoiceRecognition();
    delay(10);
}