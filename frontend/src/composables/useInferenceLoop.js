import { ref, onUnmounted } from 'vue';
import { predictImage as apiPredictImage } from '@/services/strawberryApi';

export function useInferenceLoop(videoElement, intervalMs = 500) {
  const targets = ref([]);
  const predictInFlight = ref(false);
  const isPredicting = ref(false);
  let timer = null;
  let predictAbortController = null;
  let canvasElement = null;

  const isRequestCanceled = (err) => {
    if (err === null || err === undefined) return false;
    if (typeof err !== 'object') return false;
    if ('name' in err && (err.name === 'AbortError' || err.name === 'CanceledError')) return true;
    if ('code' in err && err.code === 'ERR_CANCELED') return true;
    return false;
  };

  const stopInference = () => {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    if (predictAbortController) {
      predictAbortController.abort();
      predictAbortController = null;
    }
    predictInFlight.value = false;
    isPredicting.value = false;
  };

  const captureAndPredict = async (onSuccess = () => {}, onError = () => {}) => {
    const video = videoElement.value;
    if (!video || video.readyState < 2 || predictInFlight.value) return;

    if (!canvasElement) {
      if (typeof document === 'undefined') return;
      canvasElement = document.createElement('canvas');
    }

    const context = canvasElement.getContext('2d');
    if (!context) return;

    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);

    predictInFlight.value = true;
    isPredicting.value = true;

    canvasElement.toBlob(async (blob) => {
      if (!blob) {
        predictInFlight.value = false;
        isPredicting.value = false;
        return;
      }

      if (predictAbortController) predictAbortController.abort();
      predictAbortController = new AbortController();

      try {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        const results = await apiPredictImage(formData, predictAbortController.signal);
        targets.value = results;
        onSuccess(results);
      } catch (err) {
        if (!isRequestCanceled(err)) {
          onError(err);
        }
      } finally {
        predictInFlight.value = false;
        isPredicting.value = false;
        predictAbortController = null;
      }
    }, 'image/jpeg', 0.8);
  };

  const startInference = (onSuccess, onError) => {
    stopInference();
    timer = setInterval(() => captureAndPredict(onSuccess, onError), intervalMs);
  };

  onUnmounted(() => {
    stopInference();
  });

  return {
    targets,
    isPredicting,
    startInference,
    stopInference
  };
}
