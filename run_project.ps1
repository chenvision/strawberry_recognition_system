# Strawberry Recognition System 一键启动脚本

# 设置项目根目录
$ProjectRoot = Get-Location

Write-Host ">>> 正在初始化草莓三维姿态估计系统..." -ForegroundColor Cyan

# 1. 启动 FastAPI 后端服务
Write-Host "`n[1/2] 正在启动 FastAPI 后端服务..." -ForegroundColor Yellow
# 显式指定工作目录为项目根目录
Start-Process powershell -ArgumentList "-NoProfile -Command python app.py" -WorkingDirectory $ProjectRoot -WindowStyle Normal

# 等待后端启动（给 3 秒缓冲）
Start-Sleep -Seconds 3

# 2. 启动 Vue 前端服务
Write-Host "[2/2] 正在启动 Vue 前端服务..." -ForegroundColor Yellow
$FrontendPath = Join-Path $ProjectRoot "frontend"

if (Test-Path $FrontendPath) {
    # 检查 node_modules 是否存在
    if (-not (Test-Path (Join-Path $FrontendPath "node_modules"))) {
        Write-Host "检测到前端依赖未安装，正在执行 npm install..." -ForegroundColor Cyan
        Push-Location $FrontendPath
        npm install
        Pop-Location
    }
    # 显式指定工作目录为 frontend 目录
    Start-Process powershell -ArgumentList "-NoProfile -Command npm run serve" -WorkingDirectory $FrontendPath -WindowStyle Normal
} else {
    Write-Host "错误: 未找到 frontend 文件夹！" -ForegroundColor Red
    exit
}

Write-Host "`n>>> 系统启动指令已发出！" -ForegroundColor Green
Write-Host "-------------------------------------------"
Write-Host "后端地址: http://127.0.0.1:8000"
Write-Host "前端地址: http://localhost:8080"
Write-Host "-------------------------------------------"
Write-Host "提示: 启动后请检查弹出的两个窗口是否有报错信息。"
Read-Host "按下回车键退出此脚本..."
