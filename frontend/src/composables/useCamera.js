import { ref, onUnmounted } from 'vue';

export function useCamera(constraints = { video: { width: { ideal: 800 }, height: { ideal: 600 } } }) {
  const videoElement = ref(null);
  const isStreamActive = ref(false);
  const isCameraLoading = ref(false);
  const cameraErrorMsg = ref('');
  
  const cameraRetryCount = ref(0);
  const cameraRetryMax = 5;
  const cameraRetryBaseDelayMs = 800;
  let cameraRetryTimeoutId = null;
  let cameraAttemptInFlight = false;

  const stopStream = () => {
    if (cameraRetryTimeoutId) {
      clearTimeout(cameraRetryTimeoutId);
      cameraRetryTimeoutId = null;
    }
    cameraAttemptInFlight = false;
    isCameraLoading.value = false;

    if (videoElement.value && videoElement.value.srcObject) {
      const tracks = videoElement.value.srcObject.getTracks();
      tracks.forEach(t => t.stop());
      videoElement.value.srcObject = null;
    }
    isStreamActive.value = false;
  };

  const initCamera = async (onSuccess = () => {}, onLog = () => {}) => {
    if (cameraAttemptInFlight) return;
    cameraAttemptInFlight = true;
    isCameraLoading.value = true;
    cameraErrorMsg.value = '';

    if (cameraRetryTimeoutId) {
      clearTimeout(cameraRetryTimeoutId);
      cameraRetryTimeoutId = null;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (videoElement.value) {
        videoElement.value.srcObject = stream;
      }
      isStreamActive.value = true;
      isCameraLoading.value = false;
      cameraRetryCount.value = 0;
      onSuccess();
      onLog('摄像头已连接');
    } catch (err) {
      isStreamActive.value = false;
      cameraErrorMsg.value = err instanceof Error ? err.message : String(err);

      if (cameraRetryCount.value < cameraRetryMax) {
        const delayMs = Math.min(
          10000,
          cameraRetryBaseDelayMs * Math.pow(2, cameraRetryCount.value)
        );
        cameraRetryCount.value += 1;
        onLog(`摄像头连接失败，${Math.round(delayMs / 1000)}s 后重试 (${cameraRetryCount.value}/${cameraRetryMax})`);
        
        cameraRetryTimeoutId = setTimeout(() => {
          cameraAttemptInFlight = false;
          initCamera(onSuccess, onLog);
        }, delayMs);
        return;
      }

      isCameraLoading.value = false;
      onLog('摄像头连接失败，请检查权限或设备占用');
    } finally {
      if (!cameraRetryTimeoutId) {
        cameraAttemptInFlight = false;
      }
    }
  };

  onUnmounted(() => {
    stopStream();
  });

  return {
    videoElement,
    isStreamActive,
    isCameraLoading,
    cameraErrorMsg,
    initCamera,
    stopStream
  };
}
