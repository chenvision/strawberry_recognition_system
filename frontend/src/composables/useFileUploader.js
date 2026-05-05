import { ref, onUnmounted } from 'vue';
import { 
  analyzeImage as apiAnalyzeImage, 
  fetchDemoFrames as apiFetchDemoFrames, 
  analyzeDemoFrame as apiAnalyzeDemoFrame 
} from '@/services/strawberryApi';

export function useFileUploader() {
  const uploadLoading = ref(false);
  const analysisResult = ref({ image: '', targets: [] });
  let uploadAbortController = null;

  const demoFrames = ref([]);
  const demoResult = ref(null);
  const demoLoadingFrame = ref(null);
  let demoAbortController = null;

  const isRequestCanceled = (err) => {
    if (err === null || err === undefined) return false;
    if (typeof err !== 'object') return false;
    if ('name' in err && (err.name === 'AbortError' || err.name === 'CanceledError')) return true;
    if ('code' in err && err.code === 'ERR_CANCELED') return true;
    return false;
  };

  const analyzeFile = async (file, onLog = () => {}, onStatusChange = () => {}) => {
    if (uploadAbortController) uploadAbortController.abort();
    uploadAbortController = new AbortController();

    uploadLoading.value = true;
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await apiAnalyzeImage(formData, uploadAbortController.signal);
      analysisResult.value = { image: res.result_image, targets: res.targets };
      onLog(`上传分析完成: ${res.targets.length} 目标`);
      onStatusChange(true);
      return true;
    } catch (err) {
      if (!isRequestCanceled(err)) {
        const message = err instanceof Error ? err.message : '分析失败';
        onLog(message);
        onStatusChange(false);
      }
      return false;
    } finally {
      uploadLoading.value = false;
      uploadAbortController = null;
    }
  };

  const clearUploadResult = () => {
    if (uploadAbortController) uploadAbortController.abort();
    analysisResult.value = { image: '', targets: [] };
  };

  const fetchDemoFrames = async (limit = 60, onStatusChange = () => {}) => {
    try {
      const frames = await apiFetchDemoFrames(limit);
      demoFrames.value = frames;
      onStatusChange(true);
    } catch {
      onStatusChange(false);
    }
  };

  const analyzeDemoFrame = async (name, onLog = () => {}, onStatusChange = () => {}) => {
    if (demoAbortController) demoAbortController.abort();
    demoAbortController = new AbortController();
    demoLoadingFrame.value = name;

    try {
      const res = await apiAnalyzeDemoFrame(name, demoAbortController.signal);
      demoResult.value = res;
      onStatusChange(true);
    } catch (err) {
      if (!isRequestCanceled(err)) {
        const message = err instanceof Error ? err.message : '演示帧分析失败';
        onLog(message);
        onStatusChange(false);
      }
    } finally {
      demoLoadingFrame.value = null;
      demoAbortController = null;
    }
  };

  onUnmounted(() => {
    if (uploadAbortController) uploadAbortController.abort();
    if (demoAbortController) demoAbortController.abort();
  });

  return {
    uploadLoading,
    analysisResult,
    analyzeFile,
    clearUploadResult,
    demoFrames,
    demoResult,
    demoLoadingFrame,
    fetchDemoFrames,
    analyzeDemoFrame
  };
}
