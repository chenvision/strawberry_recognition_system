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
    // 1. 如果已有尝试在进行，直接忽略（防止并发调用）
    if (cameraAttemptInFlight) return;
    
    // 2. 清除之前的定时器，重置状态
    if (cameraRetryTimeoutId) {
      clearTimeout(cameraRetryTimeoutId);
      cameraRetryTimeoutId = null;
    }

    cameraAttemptInFlight = true;
    isCameraLoading.value = true;
    cameraErrorMsg.value = '';

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
      const errorName = err instanceof Error ? err.name : '';
      const errorMsg = err instanceof Error ? err.message : String(err);
      cameraErrorMsg.value = errorMsg;

      // 针对不同错误类型给出更具体的日志（但保持重试逻辑）
      let logPrefix = '摄像头连接失败';
      if (errorName === 'NotAllowedError' || errorName === 'PermissionDeniedError') {
        logPrefix = '摄像头访问被拒绝（请检查浏览器权限）';
      } else if (errorName === 'NotReadableError' || errorName === 'TrackStartError') {
        logPrefix = '摄像头可能被其他应用占用';
      }

      if (cameraRetryCount.value < cameraRetryMax) {
        const delayMs = Math.min(
          10000,
          cameraRetryBaseDelayMs * Math.pow(2, cameraRetryCount.value)
        );
        cameraRetryCount.value += 1;
        onLog(`${logPrefix}，${Math.round(delayMs / 1000)}s 后重试 (${cameraRetryCount.value}/${cameraRetryMax})`);
        
        cameraRetryTimeoutId = setTimeout(() => {
          cameraAttemptInFlight = false;
          initCamera(onSuccess, onLog);
        }, delayMs);
        return;
      }

      isCameraLoading.value = false;
      onLog('摄像头连接失败，建议手动刷新页面或切换至“图片分析”模式');
    } finally {
      // 只有在没有设置重试定时器的情况下，才标记尝试结束
      if (!cameraRetryTimeoutId) {
        cameraAttemptInFlight = false;
      }
    }
  };

  const manualInitCamera = (onSuccess, onLog) => {
    cameraRetryCount.value = 0; // 手动重置计数
    initCamera(onSuccess, onLog);
  };

  onUnmounted(() => {
    stopStream();
  });

  return {
    videoElement,
    isStreamActive,
    isCameraLoading,
    cameraErrorMsg,
    initCamera: manualInitCamera,
    stopStream
  };
}
