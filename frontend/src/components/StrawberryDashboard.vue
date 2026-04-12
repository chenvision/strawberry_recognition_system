<template>
  <div class="dashboard">
    <!-- ==================== Sidebar (Mode Switcher) ==================== -->
    <nav class="sidebar">
      <div class="sidebar-top">
        <div class="logo">
          <svg class="logo-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
            <path d="M12 2C12 2 8 6 8 10C8 14 12 18 12 22C12 18 16 14 16 10C16 6 12 2 12 2Z" />
          </svg>
        </div>
      </div>
      <div class="sidebar-menu">
        <button
          v-for="tab in tabs"
          :key="tab.key"
          class="nav-item"
          :class="{ active: activeTab === tab.key }"
          @click="switchTab(tab.key)"
          :title="tab.label"
        >
          <svg v-if="tab.key === 'camera'" class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><circle cx="12" cy="13" r="4"/>
          </svg>
          <svg v-if="tab.key === 'upload'" class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
          </svg>
          <svg v-if="tab.key === 'demo'" class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="2" y="2" width="20" height="20" rx="2"/><path d="M7 2v20M17 2v20M2 12h20"/>
          </svg>
          <span class="nav-label">{{ tab.label }}</span>
        </button>
      </div>
      <div class="sidebar-bottom">
        <div class="status-indicator" :class="{ active: backendOnline }" title="后端连接状态"></div>
      </div>
    </nav>

    <!-- ==================== Main Content Area ==================== -->
    <main class="main-layout">
      <!-- Top Header Bar -->
      <header class="top-bar">
        <div class="top-bar-left">
          <h1 class="project-title">Straw6D <small>草莓6D位姿估计</small></h1>
        </div>
        <div class="top-bar-right">
          <div class="system-stats">
            <span class="stat-item"><i class="dot success"></i> 后端: {{ backendOnline ? '已连接' : '断开' }}</span>
            <span class="stat-item"><i class="dot" :class="isStreamActive ? 'success' : 'warn'"></i> 相机: {{ isStreamActive ? '就绪' : '未就绪' }}</span>
          </div>
        </div>
      </header>

      <!-- Stage: Vision Panel -->
      <section class="stage-area">
        <div class="stage-container">
          <!-- Camera Mode -->
          <div v-show="activeTab === 'camera'" class="view-panel">
            <div class="viewport">
              <video v-show="isStreamActive" ref="videoElement" autoplay playsinline muted class="source-feed"></video>
              <canvas ref="canvasElement" style="display:none"></canvas>
              
              <!-- Camera Error / Placeholder -->
              <div v-if="!isStreamActive" class="camera-placeholder">
                <div class="icon-pulse">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" width="64">
                    <path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><line x1="1" y1="1" x2="23" y2="23" />
                  </svg>
                </div>
                <h3>未检测到可用摄像头</h3>
                <p>您的硬件检测正常，但访问被拒绝。请按照以下步骤排查：</p>
                <div class="troubleshoot-list">
                  <div class="tip">1. 检查地址栏左侧<b>锁头图标</b>，确认权限已设为<b>“允许”</b></div>
                  <div class="tip">2. 进入 Windows 设置 → 隐私 → 摄像头，开启<b>“允许应用访问”</b></div>
                  <div class="tip">3. 确认摄像头没有被<b>微信、会议软件</b>或其他浏览器窗口占用</div>
                </div>
                <div class="action-row">
                  <button class="btn-primary" @click="initCamera" :disabled="isCameraLoading">
                    {{ isCameraLoading ? '正在重试...' : '重新连接摄像头' }}
                  </button>
                  <button class="btn-outline" @click="activeTab = 'upload'">切换到图片上传</button>
                </div>
              </div>

              <div class="hud-overlay" v-if="isStreamActive">
                <div v-for="(target, index) in targets" :key="index" 
                     class="detection-box"
                     :style="{ left: target.center_2d[0] + 'px', top: target.center_2d[1] + 'px' }">
                  <span class="box-label">#{{ index + 1 }}</span>
                </div>
              </div>
            </div>
            <div class="view-controls">
              <span class="fps-counter" :class="{ 'text-error': !isStreamActive }">{{ isStreamActive ? '2 FPS' : 'OFFLINE' }}</span>
              <span class="resolution">{{ isStreamActive ? '800x600' : 'N/A' }}</span>
            </div>
          </div>

          <!-- Upload Mode -->
          <div v-show="activeTab === 'upload'" class="view-panel">
            <div class="upload-canvas" 
                 :class="{ 'drag-over': isDragging }"
                 @drop.prevent="handleDrop"
                 @dragover.prevent="isDragging = true"
                 @dragleave="isDragging = false">
              <template v-if="analysisResult.image && showResultInline">
                <img :src="analysisResult.image" class="result-display" />
                <button class="btn-reset" @click="clearResult">重新分析</button>
              </template>
              <div v-else class="upload-placeholder">
                <div class="icon-circle">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" width="48">
                    <path d="M12 16V8M8 12l4-4 4 4M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                  </svg>
                </div>
                <h3>点击或拖拽上传图片</h3>
                <p>支持 JPG, PNG 格式</p>
                <button class="btn-primary" @click="triggerUpload" :disabled="loading">
                  {{ loading ? '分析中...' : '选择图片' }}
                </button>
                <input type="file" ref="fileInput" @change="handleFileUpload" style="display:none">
              </div>
            </div>
          </div>

          <!-- Demo Mode -->
          <div v-show="activeTab === 'demo'" class="view-panel">
             <div v-if="demoResult" class="viewport">
                <img :src="demoResult.result_image" class="source-feed" />
                <button class="btn-reset-abs" @click="demoResult = null">返回图库</button>
             </div>
             <div v-else class="gallery-wrapper">
                <div class="gallery-grid">
                   <div v-for="frame in demoFrames" :key="frame.name" 
                        class="gallery-thumb" @click="analyzeDemoFrame(frame.name)">
                      <img :src="'http://127.0.0.1:8000' + frame.url" loading="lazy" />
                      <div class="thumb-overlay">
                        <span v-if="demoLoadingFrame === frame.name" class="loader-sm"></span>
                        <span v-else>{{ frame.name }}</span>
                      </div>
                   </div>
                </div>
             </div>
          </div>
        </div>
      </section>
    </main>

    <!-- ==================== Data Panel (Right Sidebar) ==================== -->
    <aside class="info-panel">
      <!-- Target Summary -->
      <section class="widget">
        <header class="widget-header">
          <h4>检测目标 ({{ displayTargets.length }})</h4>
        </header>
        <div class="widget-content scroll">
          <div v-if="displayTargets.length === 0" class="empty-hint">等待数据输入...</div>
          <div v-for="(target, index) in displayTargets" :key="index"
               class="data-card" :class="{ active: selectedTargetIndex === index }"
               @click="selectedTargetIndex = index">
            <div class="card-row">
              <span class="id">Target #{{ index + 1 }}</span>
              <span class="conf">{{ (target.confidence * 100).toFixed(1) }}%</span>
            </div>
            <div class="progress-bg"><div class="progress-bar" :style="{width: (target.confidence * 100)+'%'}"></div></div>
          </div>
        </div>
      </section>

      <!-- Target Detail -->
      <section class="widget" v-if="selectedTarget">
        <header class="widget-header">
          <h4>坐标与物理参数</h4>
        </header>
        <div class="widget-content">
          <div class="detail-grid">
            <div class="detail-item">
              <label>X (mm)</label>
              <div class="val">{{ selectedTarget.position.x.toFixed(1) }}</div>
            </div>
            <div class="detail-item">
              <label>Y (mm)</label>
              <div class="val">{{ selectedTarget.position.y.toFixed(1) }}</div>
            </div>
            <div class="detail-item">
              <label>Z (mm)</label>
              <div class="val">{{ selectedTarget.position.z.toFixed(1) }}</div>
            </div>
          </div>
          <div class="detail-grid dims">
            <div class="detail-item">
              <label>长度</label>
              <div class="val">{{ selectedTarget.dimensions.l.toFixed(1) }}mm</div>
            </div>
            <div class="detail-item">
              <label>宽度</label>
              <div class="val">{{ selectedTarget.dimensions.w.toFixed(1) }}mm</div>
            </div>
          </div>
        </div>
      </section>

      <!-- Logs -->
      <section class="widget log-widget">
        <header class="widget-header">
          <h4>系统事件</h4>
        </header>
        <div class="widget-content scroll">
          <div v-for="(log, i) in logs" :key="i" class="log-entry">
            <span class="time">{{ log.time }}</span>
            <span class="msg">{{ log.message }}</span>
          </div>
        </div>
      </section>
    </aside>
  </div>
</template>

<script>
/* (Keep existing script logic - no changes needed to methods) */
import axios from 'axios';

const API_BASE = 'http://127.0.0.1:8000';

export default {
  name: 'StrawberryDashboard',
  data() {
    return {
      activeTab: 'upload',
      tabs: [
        { key: 'camera', label: '实时检测' },
        { key: 'upload', label: '图片分析' },
        { key: 'demo', label: '演示图库' },
      ],
      targets: [],
      isStreamActive: false,
      isCameraLoading: false,
      cameraErrorMsg: '',
      timer: null,
      loading: false,
      isDragging: false,
      showResultInline: false,
      analysisResult: { image: '', targets: [] },
      demoFrames: [],
      demoResult: null,
      demoLoadingFrame: null,
      logs: [],
      selectedTargetIndex: 0,
      backendOnline: false,
    };
  },
  computed: {
    displayTargets() {
      if (this.activeTab === 'camera') return this.targets;
      if (this.activeTab === 'upload') return this.analysisResult.targets;
      if (this.activeTab === 'demo' && this.demoResult) return this.demoResult.targets || [];
      return [];
    },
    selectedTarget() {
      if (this.displayTargets.length > 0 && this.selectedTargetIndex < this.displayTargets.length) {
        return this.displayTargets[this.selectedTargetIndex];
      }
      return null;
    }
  },
  watch: {
    displayTargets() { this.selectedTargetIndex = 0; }
  },
  mounted() {
    this.addLog('系统就绪');
    this.checkBackend();
    this.fetchDemoFrames();
  },
  beforeUnmount() { this.stopStream(); },
  methods: {
    addLog(message) {
      const time = new Date().toLocaleTimeString('zh-CN', { hour12: false });
      this.logs.unshift({ time, message });
      if (this.logs.length > 30) this.logs.pop();
    },
    async checkBackend() {
      try {
        const res = await axios.get(`${API_BASE}/api/health`);
        this.backendOnline = res.data.status === 'ok';
      } catch { this.backendOnline = false; }
    },
    switchTab(key) {
      this.activeTab = key;
      if (key === 'camera' && !this.isStreamActive) this.initCamera();
      if (key !== 'camera') this.stopStream();
    },
    async initCamera() {
      if (this.isCameraLoading) return;
      this.isCameraLoading = true;
      this.isStreamActive = false;

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        this.addLog('错误: 浏览器不支持媒体访问(需使用localhost)');
        this.isCameraLoading = false;
        return;
      }

      this.addLog('正在唤醒相机硬件...');
      
      // 先清理旧流
      this.stopStream();

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = this.$refs.videoElement;
        
        if (video) {
          video.srcObject = stream;
          video.onloadedmetadata = () => {
            video.play();
            this.isStreamActive = true;
            this.startInferenceLoop();
            this.addLog('摄像头已启动并显示画面');
            this.isCameraLoading = false;
          };
        }
      } catch (err) {
        this.addLog(`相机访问受阻: ${err.name}`);
        this.isCameraLoading = false;
        
        // 探测具体原因
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const hasVideo = devices.some(d => d.kind === 'videoinput');
          if (!hasVideo) {
            this.addLog('诊断: 物理层未发现设备，请检查驱动或开关');
          } else {
            this.addLog('诊断: 发现设备但无法连接，请检查权限或占用');
          }
        } catch (e) {
          this.addLog('硬件层探测受限');
        }
      }
    },
    stopStream() {
      if (this.timer) clearInterval(this.timer);
      const video = this.$refs.videoElement;
      if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
      }
      this.isStreamActive = false;
    },
    startInferenceLoop() {
      this.timer = setInterval(() => this.captureAndPredict(), 500);
    },
    async captureAndPredict() {
      if (!this.isStreamActive || this.loading) return;
      const video = this.$refs.videoElement;
      const canvas = this.$refs.canvasElement;
      const context = canvas.getContext('2d');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        try {
          const res = await axios.post(`${API_BASE}/api/predict`, formData);
          this.targets = Array.isArray(res.data) ? res.data : [res.data];
        } catch (e) { /* silent fail for real-time */ }
      }, 'image/jpeg', 0.8);
    },
    triggerUpload() { this.$refs.fileInput.click(); },
    handleDrop(event) {
      this.isDragging = false;
      const file = event.dataTransfer.files[0];
      if (file) this.analyzeFile(file);
    },
    async handleFileUpload(event) {
      const file = event.target.files[0];
      if (file) await this.analyzeFile(file);
    },
    async analyzeFile(file) {
      this.loading = true;
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await axios.post(`${API_BASE}/api/analyze_image`, formData);
        this.analysisResult = { image: res.data.result_image, targets: res.data.targets };
        this.showResultInline = true;
        this.addLog(`上传分析完成: ${res.data.targets.length} 目标`);
      } catch (err) { this.addLog('分析失败'); }
      finally { this.loading = false; }
    },
    clearResult() { this.analysisResult = { image: '', targets: [] }; this.showResultInline = false; },
    async fetchDemoFrames() {
      try {
        const res = await axios.get(`${API_BASE}/api/demo/frames?limit=60`);
        this.demoFrames = res.data.frames || [];
      } catch { /* ... */ }
    },
    async analyzeDemoFrame(name) {
      this.demoLoadingFrame = name;
      try {
        const res = await axios.get(`${API_BASE}/api/demo/analyze_frame?name=${encodeURIComponent(name)}`);
        this.demoResult = res.data;
      } finally { this.demoLoadingFrame = null; }
    }
  }
};
</script>

<style scoped>
/* ===== Variables ===== */
:root {
  --sidebar-w: 64px;
  --panel-w: 320px;
  --primary: #047857;
  --bg: #f8fafc;
  --border: #e2e8f0;
  --text-main: #0f172a;
  --text-dim: #64748b;
}

/* ===== Base Layout ===== */
.dashboard {
  display: flex;
  height: 100vh;
  width: 100%;
  background-color: #f1f5f9;
  font-family: 'Inter', system-ui, sans-serif;
  color: #0f172a;
}

/* ===== Sidebar ===== */
.sidebar {
  width: 64px;
  background: #1e293b;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 16px 0;
  z-index: 10;
}

.logo-icon { width: 32px; color: #10b981; margin-bottom: 32px; }

.sidebar-menu { flex: 1; display: flex; flex-direction: column; gap: 16px; }

.nav-item {
  width: 44px;
  height: 44px;
  border-radius: 10px;
  border: none;
  background: transparent;
  color: #94a3b8;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  position: relative;
  transition: all 0.2s;
}

.nav-item:hover { background: rgba(255,255,255,0.1); color: #fff; }
.nav-item.active { background: #047857; color: #fff; box-shadow: 0 4px 12px rgba(4,120,87,0.4); }

.nav-label { display: none; } /* Hide labels in slim sidebar */

.status-indicator { width: 8px; height: 8px; border-radius: 50%; background: #ef4444; }
.status-indicator.active { background: #10b981; box-shadow: 0 0 8px #10b981; }

/* ===== Main Layout ===== */
.main-layout {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.top-bar {
  height: 56px;
  background: #fff;
  border-bottom: 1px solid #e2e8f0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
}

.project-title { font-size: 18px; font-weight: 800; margin: 0; }
.project-title small { font-size: 12px; color: #64748b; font-weight: 400; margin-left: 8px; }

.stat-item { font-size: 12px; font-weight: 600; color: #475569; margin-left: 16px; display: inline-flex; align-items: center; }
.dot { width: 6px; height: 6px; border-radius: 50%; background: #94a3b8; margin-right: 6px; }
.dot.success { background: #10b981; }
.dot.warn { background: #f59e0b; }

/* ===== Stage Area ===== */
.stage-area {
  flex: 1;
  padding: 24px;
  overflow: hidden;
  display: flex;
}

.stage-container {
  flex: 1;
  background: #fff;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.view-panel { flex: 1; display: flex; flex-direction: column; position: relative; }

.viewport {
  flex: 1;
  background: #000;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}

.source-feed { max-width: 100%; max-height: 100%; }

.hud-overlay { position: absolute; inset: 0; pointer-events: none; }

.detection-box {
  position: absolute;
  width: 40px; height: 40px;
  border: 2px solid #ef4444;
  transform: translate(-50%, -50%);
}

.box-label { background: #ef4444; color: #fff; font-size: 10px; padding: 2px 4px; position: absolute; top: -18px; left: -2px; }

/* ===== Camera Placeholder ===== */
.camera-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #64748b;
  text-align: center;
  padding: 40px;
}

.icon-pulse {
  margin-bottom: 24px;
  color: #94a3b8;
  animation: pulse-soft 2s infinite ease-in-out;
}

@keyframes pulse-soft {
  0%, 100% { transform: scale(1); opacity: 0.6; }
  50% { transform: scale(1.05); opacity: 1; }
}

.camera-placeholder h3 { font-size: 18px; color: #1e293b; margin-bottom: 8px; font-weight: 800; }
.camera-placeholder p { font-size: 13px; margin-bottom: 16px; max-width: 320px; line-height: 1.6; }

.troubleshoot-list { background: #f1f5f9; padding: 12px; border-radius: 8px; margin-bottom: 24px; text-align: left; border: 1px solid #e2e8f0; }
.tip { font-size: 11px; color: #475569; margin-bottom: 6px; line-height: 1.4; }
.tip b { color: #1e293b; }
.tip:last-child { margin-bottom: 0; }

.action-row { display: flex; gap: 12px; justify-content: center; }

.btn-primary:disabled { background: #94a3b8; cursor: not-allowed; box-shadow: none; }

.btn-outline {
  background: #fff;
  border: 1px solid #cbd5e1;
  color: #475569;
  padding: 8px 20px;
  border-radius: 8px;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-outline:hover {
  background: #f8fafc;
  border-color: #94a3b8;
  color: #1e293b;
}

.text-error { color: #ef4444; }
.upload-canvas { flex: 1; display: flex; align-items: center; justify-content: center; background: #f8fafc; position: relative; }
.upload-placeholder { text-align: center; }
.icon-circle { width: 80px; height: 80px; border-radius: 50%; background: #fff; border: 2px dashed #cbd5e1; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px; color: #94a3b8; }
.btn-primary { background: #047857; color: #fff; border: none; padding: 10px 24px; border-radius: 8px; font-weight: 700; cursor: pointer; margin-top: 16px; }
.result-display { max-width: 100%; max-height: 100%; object-fit: contain; }
.btn-reset { position: absolute; top: 16px; right: 16px; background: rgba(0,0,0,0.6); color: #fff; border: none; padding: 6px 12px; border-radius: 6px; }

/* ===== Info Panel ===== */
.info-panel {
  width: 320px;
  background: #fff;
  border-left: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
}

.widget { border-bottom: 1px solid #f1f5f9; padding: 16px; }
.widget-header h4 { margin: 0 0 12px 0; font-size: 13px; font-weight: 800; color: #1e293b; text-transform: uppercase; letter-spacing: 0.5px; }

.scroll { max-height: 200px; overflow-y: auto; }

.data-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; margin-bottom: 8px; cursor: pointer; }
.data-card.active { border-color: #047857; background: #ecfdf5; }
.card-row { display: flex; justify-content: space-between; margin-bottom: 6px; }
.card-row .id { font-weight: 700; font-size: 12px; }
.card-row .conf { font-weight: 800; color: #059669; font-size: 12px; }
.progress-bg { height: 4px; background: #e2e8f0; border-radius: 2px; }
.progress-bar { height: 100%; background: #10b981; border-radius: 2px; }

.detail-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 8px; }
.detail-item { background: #f8fafc; padding: 8px; border-radius: 6px; text-align: center; }
.detail-item label { display: block; font-size: 10px; color: #64748b; font-weight: 600; margin-bottom: 4px; }
.detail-item .val { font-size: 14px; font-weight: 800; color: #0f172a; }

.log-widget { flex: 1; display: flex; flex-direction: column; }
.log-entry { font-size: 11px; padding: 4px 0; border-bottom: 1px solid #f8fafc; display: flex; gap: 8px; }
.log-entry .time { color: #047857; font-weight: 700; }
.log-entry .msg { color: #475569; }

/* ===== Gallery ===== */
.gallery-wrapper { padding: 16px; overflow-y: auto; }
.gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 12px; }
.gallery-thumb { aspect-ratio: 1; border-radius: 8px; overflow: hidden; cursor: pointer; position: relative; border: 1px solid #e2e8f0; }
.gallery-thumb img { width: 100%; height: 100%; object-fit: cover; }
.thumb-overlay { position: absolute; inset: 0; background: rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; color: #fff; font-size: 10px; opacity: 0; transition: 0.2s; }
.gallery-thumb:hover .thumb-overlay { opacity: 1; }
</style>

