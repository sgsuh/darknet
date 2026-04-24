#!/usr/bin/env bash
# Runs the YOLO tutorial from https://pjreddie.com/darknet/yolo/
# Usage: scripts/run_yolo_tutorial.sh <command>
#   detect       YOLOv3 detection on data/dog.jpg
#   detect-ext   Same, via extended `detector test` form
#   detect-thresh YOLOv3 detection with -thresh 0 (show low-confidence boxes)
#   multi        YOLOv3 interactive multi-image mode (Ctrl-C to exit)
#   tiny         Tiny YOLOv3 detection on data/dog.jpg
#   openimages   YOLOv3 trained on Open Images
#   webcam       Webcam demo (requires OPENCV=1 build)
#   voc-data     Download + prepare Pascal VOC dataset
#   voc-train    Train YOLOv3 on VOC (requires voc-data)
#   coco-data    Download COCO dataset
#   coco-train   Train YOLOv3 on COCO (requires coco-data)
#   quick        Runs detect, detect-thresh, tiny, openimages in sequence
#   help         Show this help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DARKNET_BIN="./darknet"
WEIGHTS_BASE="https://data.pjreddie.com/files"
MEDIA_BASE="https://pjreddie.com/media/files"

log()  { printf '\n\033[1;34m[tutorial]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[tutorial]\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[1;31m[tutorial]\033[0m %s\n' "$*" >&2; exit 1; }

require_binary() {
    [[ -x "$DARKNET_BIN" ]] || die "darknet binary not found. Build first with: make -j\$(nproc)"
}

# Download <url> to <dest> unless <dest> already exists.
fetch() {
    local url="$1" dest="$2"
    if [[ -f "$dest" ]]; then
        log "Already present, skipping: $dest"
    else
        log "Downloading $url -> $dest"
        wget -q --show-progress -O "$dest" "$url"
    fi
}

cmd_detect() {
    require_binary
    fetch "$WEIGHTS_BASE/yolov3.weights" "yolov3.weights"
    log "Running: darknet detect cfg/yolov3.cfg yolov3.weights data/dog.jpg"
    "$DARKNET_BIN" detect cfg/yolov3.cfg yolov3.weights data/dog.jpg
    log "Output saved to predictions.jpg"
}

cmd_detect_ext() {
    require_binary
    fetch "$WEIGHTS_BASE/yolov3.weights" "yolov3.weights"
    log "Running extended form: darknet detector test cfg/coco.data cfg/yolov3.cfg ..."
    "$DARKNET_BIN" detector test cfg/coco.data cfg/yolov3.cfg yolov3.weights data/dog.jpg
}

cmd_detect_thresh() {
    require_binary
    fetch "$WEIGHTS_BASE/yolov3.weights" "yolov3.weights"
    log "Running with -thresh 0 (shows all detections)"
    "$DARKNET_BIN" detect cfg/yolov3.cfg yolov3.weights data/dog.jpg -thresh 0
}

cmd_multi() {
    require_binary
    fetch "$WEIGHTS_BASE/yolov3.weights" "yolov3.weights"
    log "Interactive mode. Enter image paths (e.g. data/eagle.jpg). Ctrl-C to exit."
    "$DARKNET_BIN" detect cfg/yolov3.cfg yolov3.weights
}

cmd_tiny() {
    require_binary
    fetch "$WEIGHTS_BASE/yolov3-tiny.weights" "yolov3-tiny.weights"
    log "Running tiny YOLOv3"
    "$DARKNET_BIN" detect cfg/yolov3-tiny.cfg yolov3-tiny.weights data/dog.jpg
}

cmd_openimages() {
    require_binary
    fetch "$WEIGHTS_BASE/yolov3-openimages.weights" "yolov3-openimages.weights"
    log "Running Open Images model (interactive - Ctrl-C to exit)"
    "$DARKNET_BIN" detector test cfg/openimages.data cfg/yolov3-openimages.cfg yolov3-openimages.weights
}

cmd_webcam() {
    require_binary
    if ! "$DARKNET_BIN" 2>&1 | head -1 | grep -qi .; then :; fi
    # Heuristic: we cannot introspect build flags at runtime cleanly, just warn.
    warn "Webcam demo requires Makefile OPENCV=1 and GPU=1. If not built with those, this will fail."
    fetch "$WEIGHTS_BASE/yolov3.weights" "yolov3.weights"
    log "Launching webcam demo (press Ctrl-C to stop)"
    "$DARKNET_BIN" detector demo cfg/coco.data cfg/yolov3.cfg yolov3.weights
}

cmd_voc_data() {
    log "Preparing Pascal VOC data under voc_data/"
    mkdir -p voc_data
    pushd voc_data >/dev/null
    fetch "$MEDIA_BASE/VOCtrainval_11-May-2012.tar" "VOCtrainval_11-May-2012.tar"
    fetch "$MEDIA_BASE/VOCtrainval_06-Nov-2007.tar" "VOCtrainval_06-Nov-2007.tar"
    fetch "$MEDIA_BASE/VOCtest_06-Nov-2007.tar"     "VOCtest_06-Nov-2007.tar"

    for t in VOCtrainval_11-May-2012.tar VOCtrainval_06-Nov-2007.tar VOCtest_06-Nov-2007.tar; do
        log "Extracting $t"
        tar xf "$t"
    done

    fetch "$MEDIA_BASE/voc_label.py" "voc_label.py"
    log "Generating label files"
    python3 voc_label.py

    log "Combining training sets -> train.txt"
    cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
    popd >/dev/null
    log "VOC data ready in $REPO_ROOT/voc_data/"
    warn "You must update cfg/voc.data paths to point to voc_data/ before training."
}

cmd_voc_train() {
    require_binary
    [[ -d voc_data ]] || die "Run 'voc-data' first."
    fetch "$WEIGHTS_BASE/darknet53.conv.74" "darknet53.conv.74"
    log "Training YOLOv3 on VOC (long-running)"
    "$DARKNET_BIN" detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74
}

cmd_coco_data() {
    log "Fetching COCO dataset via scripts/get_coco_dataset.sh"
    cp scripts/get_coco_dataset.sh data/
    pushd data >/dev/null
    bash get_coco_dataset.sh
    popd >/dev/null
    log "COCO data ready in $REPO_ROOT/data/coco"
}

cmd_coco_train() {
    require_binary
    [[ -d data/coco ]] || die "Run 'coco-data' first."
    fetch "$WEIGHTS_BASE/darknet53.conv.74" "darknet53.conv.74"
    log "Training YOLOv3 on COCO (long-running). Add -gpus 0,1,... for multi-GPU."
    "$DARKNET_BIN" detector train cfg/coco.data cfg/yolov3.cfg darknet53.conv.74
}

cmd_quick() {
    cmd_detect
    cmd_detect_thresh
    cmd_tiny
    cmd_openimages
}

cmd_help() {
    sed -n '2,18p' "$0"
}

main() {
    local cmd="${1:-help}"
    case "$cmd" in
        detect)         cmd_detect ;;
        detect-ext)     cmd_detect_ext ;;
        detect-thresh)  cmd_detect_thresh ;;
        multi)          cmd_multi ;;
        tiny)           cmd_tiny ;;
        openimages)     cmd_openimages ;;
        webcam)         cmd_webcam ;;
        voc-data)       cmd_voc_data ;;
        voc-train)      cmd_voc_train ;;
        coco-data)      cmd_coco_data ;;
        coco-train)     cmd_coco_train ;;
        quick)          cmd_quick ;;
        help|-h|--help) cmd_help ;;
        *) warn "Unknown command: $cmd"; cmd_help; exit 1 ;;
    esac
}

main "$@"
