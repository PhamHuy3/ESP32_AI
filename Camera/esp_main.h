#define CONFIG_CAMERA_MODULE_ESP_S3_EYE true
#define CLI_ONLY_INFERENCE 0
#define COLLECT_CPU_STATS 1
#if !defined(CLI_ONLY_INFERENCE)
#define DISPLAY_SUPPORT 0
#endif

#ifdef __cplusplus
extern "C" {
#endif
extern void run_inference(void *ptr);
#ifdef __cplusplus
}
#endif