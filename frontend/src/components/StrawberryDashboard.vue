<template>
  <div class="dashboard-container">
    <header class="dashboard-header">
      <div class="header-content">
        <h1 class="title">草莓机器人 <span class="highlight">控制台</span></h1>
        <div class="status-bar">
          <span class="status-item">系统状态: <span class="status-online">在线</span></span>
          <span class="status-item">网络连接: <span class="status-stable">稳定</span></span>
          <span class="status-item">作业模式: <span class="status-auto">自动采摘</span></span>
        </div>
      </div>
    </header>

    <main class="dashboard-content">
      <!-- Left Panel: Real-time Video Feed -->
      <section class="panel video-panel">
        <div class="panel-header">
          <h2 class="panel-title">/// 实时视觉反馈</h2>
          <div class="panel-controls">
            <!-- Upload Button -->
            <button class="upload-btn" @click="triggerUpload" title="分析静态图像">
              <span class="icon">📷</span> 分析图像
            </button>
            <input type="file" ref="fileInput" accept="image/*" style="display: none" @change="handleFileUpload">
            
            <span class="separator">|</span>
            <span class="control-indicator active">RGB 彩图</span>
            <span class="control-indicator">深度图</span>
            <span class="rec-dot"></span>
          </div>
        </div>
        <div class="video-container" v-loading="loading" element-loading-text="正在分析..." element-loading-background="rgba(0, 0, 0, 0.8)">
          <!-- Real Video Stream -->
          <div class="video-wrapper">
            <video ref="videoElement" autoplay playsinline muted class="real-video"></video>
            <canvas ref="canvasElement" class="hidden-canvas"></canvas>
            
            <!-- Overlay UI -->
            <div class="overlay-ui">
              <div class="crosshair"></div>
              <div class="scan-line"></div>
              
              <!-- Multi-Object Bounding Boxes -->
              <div v-if="targets.length > 0">
                <div v-for="(target, index) in targets" :key="index" class="target-box" 
                     :style="{ left: target.center_2d[0] + 'px', top: target.center_2d[1] + 'px' }">
                  <div class="corner top-left"></div>
                  <div class="corner top-right"></div>
                  <div class="corner bottom-left"></div>
                  <div class="corner bottom-right"></div>
                  <span class="target-label">目标 #{{ index + 1 }} [{{ (target.confidence * 100).toFixed(1) }}%]</span>
                </div>
              </div>
              <p class="no-signal" v-else>正在搜索目标...</p>
            </div>
          </div>
        </div>
        <div class="panel-footer">
          <span class="footer-info">相机 ID: STRW-CAM-01</span>
          <span class="footer-info">分辨率: 800x600 @ 30FPS</span>
        </div>
      </section>

      <!-- Right Panel: Data Visualization -->
      <section class="panel data-panel">
        <div class="panel-header">
          <h2 class="panel-title">/// 遥测数据 (主目标)</h2>
        </div>
        
        <div class="data-grid" v-if="primaryTarget">
          <!-- Confidence Score -->
          <div class="data-card confidence-card">
            <h3 class="data-label">置信度得分</h3>
            <div class="data-value-large" :class="confidenceColor">
              {{ (primaryTarget.confidence * 100).toFixed(1) }}<span class="unit">%</span>
            </div>
            <div class="progress-bar-bg">
              <div class="progress-bar-fill" :style="{ width: (primaryTarget.confidence * 100) + '%' }"></div>
            </div>
          </div>

          <!-- 3D Coordinates -->
          <div class="data-card coords-card">
            <h3 class="data-label">空间三维坐标 (3D Coordinates)</h3>
            <div class="coord-row">
              <span class="axis-label">X 轴:</span>
              <span class="axis-value">{{ primaryTarget.position.x.toFixed(2) }} <span class="unit">mm</span></span>
            </div>
            <div class="coord-row">
              <span class="axis-label">Y 轴:</span>
              <span class="axis-value">{{ primaryTarget.position.y.toFixed(2) }} <span class="unit">mm</span></span>
            </div>
            <div class="coord-row">
              <span class="axis-label">Z 轴 (深度):</span>
              <span class="axis-value">{{ primaryTarget.position.z.toFixed(2) }} <span class="unit">mm</span></span>
            </div>
          </div>

          <!-- 3D Dimensions -->
          <div class="data-card dims-card">
            <h3 class="data-label">物体物理尺寸 (Object Dimensions)</h3>
            <div class="dim-grid">
              <div class="dim-item">
                <span class="dim-label">长度</span>
                <span class="dim-value">{{ primaryTarget.dimensions.l.toFixed(1) }}</span>
                <span class="unit-small">mm</span>
              </div>
              <div class="dim-item">
                <span class="dim-label">宽度</span>
                <span class="dim-value">{{ primaryTarget.dimensions.w.toFixed(1) }}</span>
                <span class="unit-small">mm</span>
              </div>
              <div class="dim-item">
                <span class="dim-label">高度</span>
                <span class="dim-value">{{ primaryTarget.dimensions.h.toFixed(1) }}</span>
                <span class="unit-small">mm</span>
              </div>
            </div>
          </div>

          <!-- System Logs -->
          <div class="data-card log-card">
            <h3 class="data-label">系统日志 (System Logs)</h3>
            <ul class="log-list" ref="logList">
              <li v-for="(log, index) in logs" :key="index" class="log-item">
                <span class="log-time">[{{ log.time }}]</span> {{ log.message }}
              </li>
            </ul>
          </div>
        </div>
        
        <div class="data-grid" v-else>
           <div class="data-card">
             <h3 class="data-label">运行状态</h3>
             <p style="color: #666; text-align: center; padding: 20px;">未检测到目标</p>
           </div>
           <!-- System Logs (Always visible) -->
          <div class="data-card log-card">
            <h3 class="data-label">系统日志 (System Logs)</h3>
            <ul class="log-list" ref="logList">
              <li v-for="(log, index) in logs" :key="index" class="log-item">
                <span class="log-time">[{{ log.time }}]</span> {{ log.message }}
              </li>
            </ul>
          </div>
        </div>
      </section>
    </main>

    <!-- Analysis Result Modal -->
    <div v-if="showResultModal" class="modal-overlay" @click.self="closeModal">
      <div class="modal-content">
        <div class="modal-header">
          <h2 class="modal-title">/// 静态图像分析结果</h2>
          <button class="close-btn" @click="closeModal">×</button>
        </div>
        <div class="modal-body">
          <div class="result-image-container">
             <img :src="analysisResult.image" alt="分析结果" class="result-image">
          </div>
          <div class="result-data-container">
            <h3 class="data-label">检测到的目标 ({{ analysisResult.targets.length }} 个)</h3>
            <div class="result-list">
               <div v-for="(target, index) in analysisResult.targets" :key="index" class="result-item">
                 <div class="result-item-header">
                   <span class="item-id">目标 #{{ index + 1 }}</span>
                   <span class="item-conf" :class="getConfClass(target.confidence)">
                     {{ (target.confidence * 100).toFixed(1) }}%
                   </span>
                 </div>
                 <div class="result-item-body">
                   <div class="result-row">
                     <span>位置 (mm):</span>
                     <span>{{ target.position.x.toFixed(0) }}, {{ target.position.y.toFixed(0) }}, {{ target.position.z.toFixed(0) }}</span>
                   </div>
                   <div class="result-row">
                     <span>尺寸 (mm):</span>
                     <span>{{ target.dimensions.l.toFixed(0) }} x {{ target.dimensions.w.toFixed(0) }} x {{ target.dimensions.h.toFixed(0) }}</span>
                   </div>
                 </div>
               </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'StrawberryDashboard',
  data() {
    return {
      targets: [], // Array to store all detected targets (Real-time)
      logs: [],
      timer: null,
      isStreamActive: false,
      loading: false, // Loading state for analysis
      showResultModal: false,
      analysisResult: {
        image: '',
        targets: []
      }
    };
  },
  computed: {
    // Select the primary target (highest confidence) for detailed telemetry
    primaryTarget() {
      if (this.targets.length > 0) {
        return this.targets[0];
      }
      return null;
    },
    confidenceColor() {
      if (!this.primaryTarget) return '';
      if (this.primaryTarget.confidence > 0.8) return 'text-success';
      if (this.primaryTarget.confidence > 0.5) return 'text-warning';
      return 'text-danger';
    }
  },
  mounted() {
    this.addLog('系统已初始化。');
    this.initCamera();
  },
  beforeUnmount() {
    this.stopStream();
  },
  methods: {
    addLog(message) {
      const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
      this.logs.unshift({ time, message });
      if (this.logs.length > 20) this.logs.pop();
    },
    
    async initCamera() {
      this.addLog('正在请求相机访问权限...');
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 800, height: 600 } 
        });
        
        const video = this.$refs.videoElement;
        video.srcObject = stream;
        video.onloadedmetadata = () => {
          video.play();
          this.isStreamActive = true;
          this.addLog('相机视频流已激活。');
          this.startInferenceLoop();
        };
      } catch (err) {
        this.addLog(`相机访问错误: ${err.message}`);
        console.error('Camera Error:', err);
      }
    },

    stopStream() {
      if (this.timer) clearInterval(this.timer);
      
      const video = this.$refs.videoElement;
      if (video && video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
      this.isStreamActive = false;
    },

    startInferenceLoop() {
      this.addLog('开始推理循环 (2 FPS)...');
      // 2 FPS = 500ms interval
      this.timer = setInterval(() => {
        this.captureAndPredict();
      }, 500);
    },

    async captureAndPredict() {
      if (!this.isStreamActive || this.loading) return; // Pause real-time inference during analysis

      const video = this.$refs.videoElement;
      const canvas = this.$refs.canvasElement;
      const context = canvas.getContext('2d');

      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      canvas.toBlob(async (blob) => {
        if (!blob) return;

        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');

        try {
          const response = await axios.post('http://127.0.0.1:8000/api/predict', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });

          if (Array.isArray(response.data)) {
             if (response.data.length > 0 && response.data[0].error) {
                 this.addLog(`API 错误: ${response.data[0].error}`);
                 this.targets = [];
             } else {
                 this.updateData(response.data);
             }
          } else if (response.data.error) {
             this.addLog(`API 错误: ${response.data.error}`);
             this.targets = [];
          } else {
             this.updateData([response.data]);
          }

        } catch (err) {
          if (Math.random() > 0.9) {
            this.addLog(`网络连接错误: ${err.message}`);
          }
        }
      }, 'image/jpeg', 0.8);
    },

    updateData(data) {
      this.targets = data;
      if (this.targets.length > 0 && Math.random() > 0.8) {
        this.addLog(`检测到 ${this.targets.length} 个目标。`);
      }
    },

    // --- Image Analysis Logic ---
    triggerUpload() {
      this.$refs.fileInput.click();
    },

    async handleFileUpload(event) {
      const file = event.target.files[0];
      if (!file) return;

      this.loading = true;
      this.addLog(`正在分析图像: ${file.name}...`);

      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await axios.post('http://127.0.0.1:8000/api/analyze_image', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });

        if (response.data.error) {
          this.addLog(`分析错误: ${response.data.error}`);
          alert(`分析失败: ${response.data.error}`);
        } else {
          this.analysisResult = {
            image: response.data.result_image,
            targets: response.data.targets
          };
          this.showResultModal = true;
          this.addLog(`分析完成。共检测到 ${response.data.targets.length} 个目标。`);
        }
      } catch (err) {
        this.addLog(`分析失败: ${err.message}`);
        alert("分析过程中发生网络错误。");
      } finally {
        this.loading = false;
        // Reset input so same file can be selected again
        event.target.value = '';
      }
    },

    closeModal() {
      this.showResultModal = false;
    },

    getConfClass(conf) {
      if (conf > 0.8) return 'text-success';
      if (conf > 0.5) return 'text-warning';
      return 'text-danger';
    }
  }
};
</script>

<style scoped>
/* 
  Industrial/Sci-Fi Theme Variables 
*/
:root {
  --bg-color: #0a0a0a;
  --panel-bg: rgba(20, 20, 30, 0.8);
  --border-color: #00ffcc; /* Cyan Neon */
  --text-primary: #e0e0e0;
  --text-secondary: #00ffcc;
  --text-dim: #666;
  --accent-color: #00ffcc;
  --warning-color: #ffaa00;
  --danger-color: #ff3333;
  --success-color: #00ff66;
  --font-mono: 'Courier New', Courier, monospace;
}

.dashboard-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100%;
  background-color: #0a0a0a;
  color: #e0e0e0;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  overflow: hidden;
  background-image: 
    linear-gradient(rgba(0, 255, 204, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 255, 204, 0.03) 1px, transparent 1px);
  background-size: 40px 40px;
}

/* Header Styles */
.dashboard-header {
  height: 60px;
  border-bottom: 2px solid #00ffcc;
  padding: 0 20px;
  display: flex;
  align-items: center;
  background: rgba(10, 10, 10, 0.9);
  box-shadow: 0 0 15px rgba(0, 255, 204, 0.2);
}

.header-content {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.title {
  font-size: 24px;
  margin: 0;
  letter-spacing: 2px;
  font-weight: 300;
}

.highlight {
  color: #00ffcc;
  font-weight: 700;
}

.status-bar {
  display: flex;
  gap: 20px;
  font-family: 'Courier New', Courier, monospace;
  font-size: 14px;
}

.status-item {
  color: #888;
}

.status-online { color: #00ff66; text-shadow: 0 0 5px #00ff66; }
.status-stable { color: #00aaff; }
.status-auto { color: #ffaa00; }

/* Main Content Layout */
.dashboard-content {
  flex: 1;
  display: flex;
  padding: 20px;
  gap: 20px;
}

.panel {
  background: rgba(20, 25, 30, 0.7);
  border: 1px solid rgba(0, 255, 204, 0.3);
  border-radius: 4px;
  display: flex;
  flex-direction: column;
  position: relative;
}

.panel::before {
  content: '';
  position: absolute;
  top: -1px; left: -1px;
  width: 20px; height: 20px;
  border-top: 2px solid #00ffcc;
  border-left: 2px solid #00ffcc;
}
.panel::after {
  content: '';
  position: absolute;
  bottom: -1px; right: -1px;
  width: 20px; height: 20px;
  border-bottom: 2px solid #00ffcc;
  border-right: 2px solid #00ffcc;
}

.panel-header {
  padding: 10px 15px;
  border-bottom: 1px solid rgba(0, 255, 204, 0.2);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(0, 255, 204, 0.05);
}

.panel-title {
  font-size: 14px;
  margin: 0;
  color: #00ffcc;
  font-family: 'Courier New', Courier, monospace;
  letter-spacing: 1px;
}

/* Video Panel */
.video-panel {
  flex: 2;
}

.video-container {
  flex: 1;
  background: #000;
  position: relative;
  overflow: hidden;
  margin: 10px;
  border: 1px solid #333;
}

.video-wrapper {
  width: 100%;
  height: 100%;
  position: relative;
  background: #000;
}

.real-video {
  width: 100%;
  height: 100%;
  object-fit: cover; /* or contain */
  display: block; /* ensure no extra space */
}

.hidden-canvas {
  display: none;
}

.overlay-ui {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.crosshair {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 40px;
  height: 40px;
  border: 1px solid rgba(0, 255, 204, 0.5);
  border-radius: 50%;
}
.crosshair::before, .crosshair::after {
  content: '';
  position: absolute;
  background: rgba(0, 255, 204, 0.5);
}
.crosshair::before { top: 19px; left: -10px; width: 60px; height: 1px; }
.crosshair::after { left: 19px; top: -10px; height: 60px; width: 1px; }

.scan-line {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 2px;
  background: rgba(0, 255, 204, 0.3);
  box-shadow: 0 0 10px rgba(0, 255, 204, 0.5);
  animation: scan 3s linear infinite;
}

@keyframes scan {
  0% { top: 0%; opacity: 0; }
  10% { opacity: 1; }
  90% { opacity: 1; }
  100% { top: 100%; opacity: 0; }
}

/* Updated Target Box for Multi-Object */
.target-box {
  position: absolute;
  transform: translate(-50%, -50%); /* Center the box on the point */
  width: 100px; /* Fixed size for visualization or could be dynamic */
  height: 80px;
  border: 1px solid rgba(255, 50, 50, 0.6);
  box-shadow: 0 0 10px rgba(255, 50, 50, 0.2);
  transition: all 0.2s ease-out; /* Smooth movement */
}

.target-label {
  position: absolute;
  top: -20px;
  left: 0;
  background: rgba(255, 50, 50, 0.8);
  color: #fff;
  font-size: 10px;
  padding: 2px 5px;
  font-family: 'Courier New', Courier, monospace;
  white-space: nowrap;
}

.no-signal {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  color: rgba(255, 50, 50, 0.8);
  font-family: 'Courier New', Courier, monospace;
  font-size: 12px;
  letter-spacing: 2px;
  animation: blink 1s infinite;
}

.panel-controls {
  display: flex;
  gap: 10px;
  align-items: center;
}

/* Upload Button Style */
.upload-btn {
  background: rgba(0, 255, 204, 0.1);
  border: 1px solid #00ffcc;
  color: #00ffcc;
  padding: 4px 10px;
  font-size: 11px;
  font-family: 'Courier New', Courier, monospace;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 5px;
  transition: all 0.2s;
}
.upload-btn:hover {
  background: rgba(0, 255, 204, 0.3);
  box-shadow: 0 0 10px rgba(0, 255, 204, 0.3);
}
.separator {
  color: #444;
  margin: 0 5px;
}

.control-indicator {
  font-size: 10px;
  padding: 2px 6px;
  border: 1px solid #444;
  color: #666;
}
.control-indicator.active {
  border-color: #00ffcc;
  color: #00ffcc;
  background: rgba(0, 255, 204, 0.1);
}

.rec-dot {
  width: 8px;
  height: 8px;
  background: #ff3333;
  border-radius: 50%;
  animation: blink 1s infinite;
}

@keyframes blink { 50% { opacity: 0.3; } }

.panel-footer {
  padding: 5px 15px;
  font-size: 10px;
  color: #666;
  font-family: 'Courier New', Courier, monospace;
  display: flex;
  justify-content: space-between;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
}

/* Data Panel */
.data-panel {
  flex: 1;
  min-width: 300px;
}

.data-grid {
  padding: 15px;
  display: flex;
  flex-direction: column;
  gap: 15px;
  overflow-y: auto;
}

.data-card {
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.1);
  padding: 15px;
  position: relative;
}

.data-label {
  font-size: 10px;
  color: #888;
  margin: 0 0 10px 0;
  letter-spacing: 1px;
}

.data-value-large {
  font-size: 36px;
  font-family: 'Courier New', Courier, monospace;
  font-weight: 700;
}

.unit { font-size: 14px; color: #666; margin-left: 5px; }
.unit-small { font-size: 10px; color: #666; }

.text-success { color: #00ff66; text-shadow: 0 0 10px rgba(0, 255, 102, 0.3); }
.text-warning { color: #ffaa00; }
.text-danger { color: #ff3333; }

.progress-bar-bg {
  height: 4px;
  background: #333;
  margin-top: 10px;
  width: 100%;
}
.progress-bar-fill {
  height: 100%;
  background: #00ffcc;
  box-shadow: 0 0 5px #00ffcc;
  transition: width 0.3s ease;
}

.coord-row {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px dashed rgba(255, 255, 255, 0.1);
  padding: 8px 0;
  font-family: 'Courier New', Courier, monospace;
}
.axis-label { color: #00ffcc; }
.axis-value { font-size: 16px; }

.dim-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
}

.dim-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  background: rgba(255, 255, 255, 0.05);
  padding: 8px;
}

.dim-label { font-size: 8px; color: #aaa; margin-bottom: 5px; }
.dim-value { font-family: 'Courier New', Courier, monospace; font-size: 14px; color: #fff; }

.log-list {
  list-style: none;
  padding: 0;
  margin: 0;
  height: 100px;
  overflow-y: auto;
  font-family: 'Courier New', Courier, monospace;
  font-size: 10px;
}

.log-item {
  margin-bottom: 4px;
  color: #aaa;
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  padding-bottom: 2px;
}

.log-time { color: #00ffcc; margin-right: 5px; }

/* Scrollbar styling */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #111; }
::-webkit-scrollbar-thumb { background: #333; }
::-webkit-scrollbar-thumb:hover { background: #555; }

/* Modal Styles */
.modal-overlay {
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  background: rgba(0, 0, 0, 0.85);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}
.modal-content {
  background: #0f0f12;
  border: 1px solid #00ffcc;
  box-shadow: 0 0 30px rgba(0, 255, 204, 0.2);
  width: 90%;
  max-width: 1000px;
  height: 80%;
  display: flex;
  flex-direction: column;
}
.modal-header {
  padding: 15px;
  border-bottom: 1px solid #333;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(0, 255, 204, 0.1);
}
.modal-title {
  margin: 0;
  font-size: 18px;
  color: #00ffcc;
  font-weight: 400;
  letter-spacing: 2px;
}
.close-btn {
  background: none;
  border: none;
  color: #00ffcc;
  font-size: 24px;
  cursor: pointer;
}
.modal-body {
  flex: 1;
  display: flex;
  overflow: hidden;
}
.result-image-container {
  flex: 2;
  background: #000;
  display: flex;
  justify-content: center;
  align-items: center;
  border-right: 1px solid #333;
}
.result-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}
.result-data-container {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  background: rgba(255, 255, 255, 0.02);
}
.result-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.result-item {
  background: rgba(0, 0, 0, 0.4);
  border: 1px solid #333;
  padding: 10px;
}
.result-item-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 5px;
  font-size: 12px;
  color: #888;
}
.item-id { color: #00ffcc; }
.result-item-body {
  font-family: 'Courier New', Courier, monospace;
  font-size: 12px;
  color: #e0e0e0;
}
.result-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 2px;
}
</style>
