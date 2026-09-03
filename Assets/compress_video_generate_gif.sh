#!/usr/bin/env bash
# ============================================================================
# 视频压缩 + GIF 生成 —— 一键自动化脚本
# ----------------------------------------------------------------------------
# 流程：原始视频 ──▶ 压缩视频(<100MB) ──▶ GIF(20~30MB)
#
# 依赖：ffmpeg（需含 libx264 与 ffprobe）
#
# 用法：
#   bash compress_video.sh [输入视频]
#   bash compress_video.sh 输入.mp4 -s 95 -t 15 -w 450
#   bash compress_video.sh 输入.mp4 --skip-video      # 仅生成 GIF（输入已是压缩视频）
#   bash compress_video.sh 输入.mp4 --skip-gif        # 仅压缩视频
#
# 不传输入视频时，自动选择当前目录下最大的 .mp4 文件。
# ============================================================================

set -euo pipefail

# ============================================================================
# 一、默认参数（可用命令行参数覆盖）
# ============================================================================

TARGET_VIDEO_MB=95        # 压缩视频目标总大小(MB)，留出余量保证 < 100MB
AUDIO_KBPS=128            # 音频码率(kbps, AAC)
PRESET="medium"           # x264 编码预设：faster/fast/medium/slow（越慢画质越好）
MAXRATE_MULTIPLIER=2      # maxrate = 目标码率 * 此倍数

GIF_FPS=15                # GIF 帧率
GIF_WIDTH=450             # GIF 输出宽度(px)，高度按源比例自动
GIF_TARGET_MB=25          # GIF 期望大小(MB)，自动微调宽度使其落在 20~30MB
PALETTE_STATS="diff"      # 调色板统计模式：full(静态图) / diff(动画)
BAYER_SCALE=5             # bayer 抖动粒度（越大文件越小、颗粒越粗）

# ============================================================================
# 二、帮助信息
# ============================================================================

usage() {
  cat <<'EOF'
用法: bash compress_video.sh [输入视频] [选项]

位置参数:
  输入视频              要处理的视频路径（不传则自动选择当前目录最大的 .mp4）

选项:
  -o, --out-video <名>  压缩视频输出文件名（默认: <输入名>_compressed.mp4）
  -g, --out-gif <名>    GIF 输出文件名（默认: <输入名>_compressed.gif）
  -s, --video-size <MB> 压缩视频目标大小，默认 95（自动反推码率，保证 <100MB）
  -t, --gif-fps <n>     GIF 帧率，默认 15
  -w, --gif-width <px>  GIF 初始宽度，默认 450（脚本会自动微调以命中 20~30MB）
      --skip-video      跳过压缩，直接对输入视频生成 GIF
      --skip-gif        只压缩视频，不生成 GIF
  -h, --help            显示本帮助

示例:
  bash compress_video.sh
  bash compress_video.sh demo.mp4
  bash compress_video.sh demo.mp4 --skip-video
  bash compress_video.sh demo.mp4 -s 92 -w 500
EOF
  exit 0
}

# ============================================================================
# 三、命令行参数解析
# ============================================================================

INPUT=""
OUT_VIDEO=""
OUT_GIF=""
SKIP_VIDEO=0
SKIP_GIF=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)            usage ;;
    -o|--out-video)       OUT_VIDEO="$2";  shift 2 ;;
    -g|--out-gif)         OUT_GIF="$2";    shift 2 ;;
    -s|--video-size)      TARGET_VIDEO_MB="$2"; shift 2 ;;
    -t|--gif-fps)         GIF_FPS="$2";    shift 2 ;;
    -w|--gif-width)       GIF_WIDTH="$2";  shift 2 ;;
    --skip-video)         SKIP_VIDEO=1;    shift ;;
    --skip-gif)           SKIP_GIF=1;      shift ;;
    -*)                   echo "未知参数: $1（用 -h 查看帮助）"; exit 2 ;;
    *)                    INPUT="$1";      shift ;;
  esac
done

# ============================================================================
# 四、前置检查与输入识别
# ============================================================================

command -v ffmpeg  >/dev/null 2>&1 || { echo "错误：未找到 ffmpeg"; exit 1; }
command -v ffprobe >/dev/null 2>&1 || { echo "错误：未找到 ffprobe"; exit 1; }

# 未指定输入时，自动选择当前目录下最大的 .mp4
if [[ -z "$INPUT" ]]; then
  INPUT=$(ls -S ./*.mp4 2>/dev/null | head -n1)
  [[ -z "$INPUT" ]] && { echo "错误：未指定输入视频，且当前目录无 .mp4 文件"; exit 1; }
  echo "自动选择输入视频: $INPUT"
fi
[ -f "$INPUT" ] || { echo "错误：找不到输入视频 $INPUT"; exit 1; }

# 由输入名推导输出名（保留目录，落在输入视频同目录下）
BASE="${INPUT%.*}"
OUT_VIDEO="${OUT_VIDEO:-${BASE}_compressed.mp4}"
OUT_GIF="${OUT_GIF:-${BASE}_compressed.gif}"

# ============================================================================
# 五、压缩视频（两遍编码，自动反推码率，目标 <100MB）
# ----------------------------------------------------------------------------
# 码率反推：总大小 = 视频 + 音频。先算音频占用，剩下的分给视频。
#   视频码率(kbps) = (目标总比特 - 音频比特) / 时长 / 1000
# 两遍编码在固定体积约束下画质更好；-maxrate/-bufsize 控制码率波动。
# ============================================================================

if [[ "$SKIP_VIDEO" -eq 1 ]]; then
  GIF_SOURCE="$INPUT"
  echo "跳过视频压缩，直接对 $INPUT 生成 GIF"
else
  DURATION=$(ffprobe -v error -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 "$INPUT")
  echo "输入视频时长：${DURATION} 秒"

  TOTAL_BITS=$(( TARGET_VIDEO_MB * 1000 * 1000 * 8 ))   # 1MB=1,000,000 字节，留余量
  AUDIO_BITS=$(awk -v ak="$AUDIO_KBPS" -v d="$DURATION" 'BEGIN{printf "%.0f", ak*1000*d}')
  VIDEO_KBPS=$(awk -v tb="$TOTAL_BITS" -v ab="$AUDIO_BITS" -v d="$DURATION" \
    'BEGIN{printf "%.0f", (tb-ab)/d/1000}')
  echo "目标总大小：${TARGET_VIDEO_MB}MB → 视频码率 ${VIDEO_KBPS}kbps"

  echo "== 压缩视频：第一遍（分析）=="
  ffmpeg -y -i "$INPUT" \
    -c:v libx264 -b:v "${VIDEO_KBPS}k" -maxrate "${VIDEO_KBPS}k" \
    -bufsize "$(( VIDEO_KBPS * MAXRATE_MULTIPLIER ))k" -preset "$PRESET" \
    -pass 1 -an -f null -

  echo "== 压缩视频：第二遍（编码）=="
  ffmpeg -y -i "$INPUT" \
    -c:v libx264 -b:v "${VIDEO_KBPS}k" -maxrate "${VIDEO_KBPS}k" \
    -bufsize "$(( VIDEO_KBPS * MAXRATE_MULTIPLIER ))k" -preset "$PRESET" \
    -pass 2 -c:a aac -b:a "${AUDIO_KBPS}k" "$OUT_VIDEO"

  rm -f ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree
  GIF_SOURCE="$OUT_VIDEO"
fi

# ============================================================================
# 六、生成 GIF（调色板 + bayer 抖动，自动微调宽度命中 20~30MB）
# ----------------------------------------------------------------------------
# 第一步：采样生成 256 色调色板（stats_mode=diff 面向动画）
# 第二步：用调色板渲染 GIF（bayer 抖动，体积比 sierra2_4a 小 2~3 倍）
# 由于 GIF 体积与画面内容强相关，生成后自动按比例缩放宽度重试，直到落在
# 20~30MB 区间（最多重试 4 次）。
# ============================================================================

if [[ "$SKIP_GIF" -eq 0 ]]; then
  gen_gif() {  # 参数: 宽度
    ffmpeg -y -v error -i "$GIF_SOURCE" \
      -vf "fps=${GIF_FPS},scale=$1:-2:flags=lanczos,palettegen=stats_mode=${PALETTE_STATS}" \
      palette.png
    ffmpeg -y -v error -i "$GIF_SOURCE" -i palette.png \
      -lavfi "fps=${GIF_FPS},scale=$1:-2:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=${BAYER_SCALE}:diff_mode=rectangle" \
      "$OUT_GIF"
    rm -f palette.png
  }

  w="$GIF_WIDTH"
  for _ in 1 2 3 4; do
    echo "== 生成 GIF（宽度 ${w}px）=="
    gen_gif "$w"
    SIZE=$(stat -c%s "$OUT_GIF")   # 字节数
    echo "    当前 GIF 大小：$(( SIZE / 1000000 ))MB"

    if (( SIZE > 30000000 )); then        # 超过 30MB → 缩小
      w=$(awk -v w="$w" -v s="$SIZE" 'BEGIN{printf "%.0f", w*sqrt(25000000/s)}')
      (( w < 200 )) && w=200
    elif (( SIZE < 20000000 )); then      # 低于 20MB → 放大
      w=$(awk -v w="$w" -v s="$SIZE" 'BEGIN{printf "%.0f", w*sqrt(25000000/s)}')
    else
      break
    fi
  done
fi

# ============================================================================
# 七、结果汇总
# ============================================================================

echo ""
echo "========== 完成 =========="
if [[ "$SKIP_VIDEO" -eq 0 ]]; then
  echo "[视频] $OUT_VIDEO  ->  $(du -h "$OUT_VIDEO" | cut -f1)"
fi
if [[ "$SKIP_GIF" -eq 0 ]]; then
  echo "[GIF ] $OUT_GIF    ->  $(du -h "$OUT_GIF" | cut -f1)"
fi
