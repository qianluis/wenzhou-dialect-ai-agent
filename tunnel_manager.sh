#!/bin/bash
# ===================================================
#  🚀 温州话隧道管理脚本
#  启动: bash tunnel_manager.sh start
#  停止: bash tunnel_manager.sh stop
#  状态: bash tunnel_manager.sh status
#  前台运行: bash tunnel_manager.sh run
# ===================================================
set -e

SERVER_DIR="/home/sandbox/.openclaw/workspace/wenzhou-project/wenzhou2mandarin-agent"
PORT=8000
URL_FILE="/tmp/wenzhou_live_url.txt"
PID_FILE="/tmp/wenzhou_manager.pid"
LOG_FILE="/tmp/wenzhou_manager.log"
SSH_LOG="/tmp/wenzhou_ssh.log"

log() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG_FILE"; echo "$*"; }

# ── 启动服务 ──
start_server() {
    log "Starting server..."
    cd "$SERVER_DIR"
    FUSION_MODE=single ASR_BACKEND=mock ASR_WENZHOU_BACKEND=mock ASR_MANDARIN_BACKEND=mock PORT=$PORT \
        nohup python3 -m uvicorn serve:app --host 0.0.0.0 --port $PORT --log-level warning \
        > /tmp/srv.log 2>&1 &
    local pid=$!
    # Wait for it
    for i in $(seq 1 15); do
        if curl -sf http://localhost:$PORT/api/lexicon/count > /dev/null 2>&1; then
            log "✅ Server up (PID $pid)"
            return 0
        fi
        sleep 1
    done
    log "❌ Server failed to start"
    return 1
}

# ── 检查服务 + 隧道 ──
check_health() {
    # 1. 本地服务
    if ! curl -sf http://localhost:$PORT/api/lexicon/count > /dev/null 2>&1; then
        log "⚠️  Local server down, restarting..."
        start_server
    fi

    # 2. 公网隧道
    local url=""
    [ -f "$URL_FILE" ] && url=$(cat "$URL_FILE")
    if [ -n "$url" ]; then
        local code=$(curl -s -o /dev/null -w "%{http_code}" "$url/" 2>&1)
        if [ "$code" = "200" ]; then
            return 0  # 一切正常
        fi
        log "⚠️  Tunnel returns $code, need restart"
    else
        log "⚠️  No URL file yet"
    fi
    return 1  # 需要重建
}

# ── 建立隧道 ──
start_tunnel() {
    # 杀掉旧隧道
    pkill -f "nokey@localhost.run" 2>/dev/null
    rm -f "$URL_FILE"
    : > "$SSH_LOG"

    log "Starting tunnel..."

    # 启动 SSH
    ssh -o StrictHostKeyChecking=no \
        -o ServerAliveInterval=15 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -R "80:localhost:$PORT" \
        nokey@localhost.run \
        > "$SSH_LOG" 2>&1 &
    local pid=$!

    # 等待获取 URL（最多 20s）
    for i in $(seq 1 20); do
        local url
        url=$(grep -oP 'https://[a-zA-Z0-9-]+\.lhr\.life' "$SSH_LOG" 2>/dev/null | tail -1)
        if [ -n "$url" ]; then
            echo "$url" > "$URL_FILE"
            log "✅ Tunnel up: $url"
            return 0
        fi
        sleep 1
    done

    log "❌ Tunnel failed to get URL"
    tail -10 "$SSH_LOG" | while read line; do log "  SSH: $line"; done
    return 1
}

# ── 主循环 ──
run_loop() {
    log "========================================"
    log "  🚀 温州话隧道管理器 启动"
    log "  端口: $PORT"
    log "  检测间隔: 20秒"
    log "========================================"

    # 启动服务
    start_server

    # 首次建立隧道
    start_tunnel

    # 循环检测
    local loop_count=0
    while true; do
        loop_count=$((loop_count + 1))
        sleep 20

        if ! check_health; then
            log "🔄 Rebuilding tunnel (attempt $loop_count)..."
            start_tunnel
        elif [ $((loop_count % 15)) -eq 0 ]; then
            # Every 5 minutes, log heartbeat
            local url=$(cat "$URL_FILE" 2>/dev/null)
            log "💚 Heartbeat - $url"
        fi
    done
}

# ── 后台启动 ──
start_daemon() {
    if [ -f "$PID_FILE" ]; then
        local old_pid=$(cat "$PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            log "Already running (PID $old_pid)"
            echo "Already running (PID $old_pid)"
            return 0
        fi
        rm -f "$PID_FILE"
    fi
    nohup bash "$0" run > /dev/null 2>&1 &
    local pid=$!
    echo "$pid" > "$PID_FILE"
    log "Daemon started (PID $pid)"
    echo "Daemon started (PID $pid)"
    sleep 3
    show_status
}

stop_daemon() {
    if [ -f "$PID_FILE" ]; then
        kill $(cat "$PID_FILE") 2>/dev/null
        rm -f "$PID_FILE"
    fi
    pkill -f "nokey@localhost.run" 2>/dev/null
    log "Stopped"
    echo "Stopped"
}

show_status() {
    local url=$(cat "$URL_FILE" 2>/dev/null)
    local code="?"
    [ -n "$url" ] && code=$(curl -s -o /dev/null -w "%{http_code}" "$url/" 2>&1)
    local srv_code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$PORT/ 2>&1)

    echo ""
    echo "  📍 公网: $url"
    echo "  🌐 公网状态: HTTP $code"
    echo "  🖥️  本地状态: HTTP $srv_code"
    echo "  📊 词典: $(curl -sf http://localhost:$PORT/api/lexicon/count 2>/dev/null)"
    echo "  🔄 循环: $(tail -5 "$LOG_FILE" 2>/dev/null | grep -c 'Heartbeat\|Tunnel up\|Rebuild') 次心跳"
    echo ""
}

# ── 入口 ──
case "${1:-start}" in
    run)   run_loop ;;
    start) start_daemon ;;
    stop)  stop_daemon ;;
    restart) stop_daemon; sleep 2; start_daemon ;;
    status) show_status ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|run}"
        exit 1
        ;;
esac
