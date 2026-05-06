import { ref, onUnmounted } from 'vue';

export function useWebSocketInference(videoElement) {
  const targets = ref([]);
  const isConnected = ref(false);
  const isPredicting = ref(false);
  const processingTime = ref(0);
  let socket = null;
  let canvasElement = null;
  let animationFrameId = null;

  const connect = () => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // 假设后端在 8000 端口，或者根据配置获取
    const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/predict`;
    
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log('WebSocket Connected');
      isConnected.value = true;
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.targets) {
        targets.value = data.targets;
        processingTime.value = data.processing_time_ms;
      }
    };

    socket.onclose = () => {
      console.log('WebSocket Disconnected');
      isConnected.value = false;
      stopInference();
    };

    socket.onerror = (error) => {
      console.error('WebSocket Error:', error);
    };
  };

  const captureAndSend = () => {
    const video = videoElement.value;
    if (!video || video.readyState < 2 || !isConnected.value) {
        animationFrameId = requestAnimationFrame(captureAndSend);
        return;
    }

    if (!canvasElement) {
      canvasElement = document.createElement('canvas');
    }

    const context = canvasElement.getContext('2d');
    canvasElement.width = 640; // 降低分辨率以提升速度
    canvasElement.height = 480;
    context.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);

    canvasElement.toBlob((blob) => {
      if (blob && socket.readyState === WebSocket.OPEN) {
        socket.send(blob);
      }
      // 这里我们可以控制发送频率，或者由后端背压控制
      // 为了演示，我们使用 requestAnimationFrame，但后端有处理锁
      animationFrameId = requestAnimationFrame(captureAndSend);
    }, 'image/jpeg', 0.7);
  };

  const startInference = () => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      connect();
    }
    isPredicting.value = true;
    captureAndSend();
  };

  const stopInference = () => {
    isPredicting.value = false;
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
      animationFrameId = null;
    }
    if (socket) {
      socket.close();
      socket = null;
    }
  };

  onUnmounted(() => {
    stopInference();
  });

  return {
    targets,
    isConnected,
    isPredicting,
    processingTime,
    startInference,
    stopInference
  };
}
