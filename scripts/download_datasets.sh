#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-./data/seg}"

usage() {
  cat <<'EOF'
Usage:
  DATA_ROOT=/path/to/seg bash scripts/download_datasets.sh <dataset> [dataset...]

Datasets:
  coco        COCOStuff 164k
  cityscapes  Cityscapes train/val/test images and gtFine labels
  voc         PASCAL VOC 2012 trainval
  ade20k      ADE20K challenge 2016
  davis       DAVIS 2017 trainval 480p
  all         All datasets above

Cityscapes requires an account:
  CITYSCAPES_USERNAME=<username> CITYSCAPES_PASSWORD=<password> DATA_ROOT=/path/to/seg \
    bash scripts/download_datasets.sh cityscapes
EOF
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

download_file() {
  local url="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  wget -nc --directory-prefix="$out_dir" "$url"
}

download_coco() {
  local root="$DATA_ROOT/COCOStuff"
  local downloads="$root/downloads"
  mkdir -p "$root/dataset/images" "$root/dataset/annotations" "$downloads"

  download_file "http://images.cocodataset.org/zips/train2017.zip" "$downloads"
  download_file "http://images.cocodataset.org/zips/val2017.zip" "$downloads"
  download_file "http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip" "$downloads"
  download_file "https://www.robots.ox.ac.uk/~xuji/datasets/COCOStuff164kCurated.tar.gz" "$downloads"

  unzip -n "$downloads/train2017.zip" -d "$root/dataset/images/"
  unzip -n "$downloads/val2017.zip" -d "$root/dataset/images/"
  unzip -n "$downloads/stuffthingmaps_trainval2017.zip" -d "$root/dataset/annotations/"

  if [[ ! -d "$root/curated" ]]; then
    tar -xzf "$downloads/COCOStuff164kCurated.tar.gz" -C "$root"
    mv "$root/COCO/COCOStuff164k" "$root/curated"
    rmdir "$root/COCO"
  fi
}

download_cityscapes() {
  : "${CITYSCAPES_USERNAME:?Set CITYSCAPES_USERNAME before downloading Cityscapes.}"
  : "${CITYSCAPES_PASSWORD:?Set CITYSCAPES_PASSWORD before downloading Cityscapes.}"

  local root="$DATA_ROOT/cityscapes"
  local downloads="$root/downloads"
  local cookies="$downloads/cookies.txt"
  mkdir -p "$downloads"

  wget \
    --keep-session-cookies \
    --save-cookies="$cookies" \
    --post-data "username=${CITYSCAPES_USERNAME}&password=${CITYSCAPES_PASSWORD}&submit=Login" \
    -O /dev/null \
    "https://www.cityscapes-dataset.com/login/"

  wget --load-cookies "$cookies" --content-disposition --directory-prefix="$downloads" \
    "https://www.cityscapes-dataset.com/file-handling/?packageID=1"
  wget --load-cookies "$cookies" --content-disposition --directory-prefix="$downloads" \
    "https://www.cityscapes-dataset.com/file-handling/?packageID=3"

  unzip -n "$downloads/gtFine_trainvaltest.zip" -d "$root"
  unzip -n "$downloads/leftImg8bit_trainvaltest.zip" -d "$root"
}

download_voc() {
  local root="$DATA_ROOT/VOC"
  local downloads="$root/downloads"
  mkdir -p "$downloads"

  download_file "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" "$downloads"
  tar -xf "$downloads/VOCtrainval_11-May-2012.tar" -C "$root"
}

download_ade20k() {
  local downloads="$DATA_ROOT/downloads"
  mkdir -p "$downloads"

  download_file "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip" "$downloads"
  unzip -n "$downloads/ADEChallengeData2016.zip" -d "$DATA_ROOT"
}

download_davis() {
  local root="$DATA_ROOT/DAVIS"
  local downloads="$root/downloads"
  mkdir -p "$downloads"

  download_file "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip" "$downloads"
  unzip -n "$downloads/DAVIS-2017-trainval-480p.zip" -d "$root"

  if [[ -d "$root/DAVIS" ]]; then
    shopt -s dotglob
    mv "$root/DAVIS"/* "$root/"
    rmdir "$root/DAVIS"
    shopt -u dotglob
  fi
}

main() {
  require_cmd wget
  require_cmd unzip
  require_cmd tar

  if [[ "$#" -eq 0 ]]; then
    usage
    exit 1
  fi

  mkdir -p "$DATA_ROOT"

  local datasets=("$@")
  if [[ " ${datasets[*]} " == *" all "* ]]; then
    datasets=(coco cityscapes voc ade20k davis)
  fi

  for dataset in "${datasets[@]}"; do
    case "$dataset" in
      coco|cocostuff) download_coco ;;
      cityscapes) download_cityscapes ;;
      voc) download_voc ;;
      ade20k) download_ade20k ;;
      davis) download_davis ;;
      -h|--help) usage ;;
      *)
        echo "Unknown dataset: $dataset" >&2
        usage
        exit 1
        ;;
    esac
  done
}

main "$@"
