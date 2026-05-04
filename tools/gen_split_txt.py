"""
gen_split_txt.py
================
Tự động tạo các file *.txt chứa đường dẫn ảnh (chuẩn YOLO) từ tên video.

Cách dùng:
----------
# Gen val.txt + val2.txt cho các video phantom02, phantom05, phantom08
python tools/gen_split_txt.py --split val --videos phantom02 phantom05 phantom08

# Gen train.txt + train2.txt cho toàn bộ video trừ danh sách val và test
python tools/gen_split_txt.py --split train --exclude phantom02 phantom05 phantom08

# Gen test.txt + test2.txt
python tools/gen_split_txt.py --split test --videos phantom02 phantom05
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # YOLOMG root
IMG_DIR   = ROOT / 'datasets' / 'images'
MASK_DIR  = ROOT / 'datasets' / 'ARD100_mask32'
OUT_DIR   = ROOT / 'datasets'


def gen_txt(videos, split):
    """Write <split>.txt (images) and <split>2.txt (masks)."""
    img_lines, mask_lines = [], []

    for vid in sorted(videos):
        img_folder  = IMG_DIR  / vid
        mask_folder = MASK_DIR / vid

        imgs  = sorted(img_folder.glob('*.jpg'))  if img_folder.exists()  else []
        masks = sorted(mask_folder.glob('*.jpg')) if mask_folder.exists() else []

        # Clip to shortest so they stay in sync
        n = min(len(imgs), len(masks))
        if n == 0:
            print(f'  [WARN] {vid}: no images/masks found, skipping.')
            continue
        if len(imgs) != len(masks):
            print(f'  [WARN] {vid}: images={len(imgs)}, masks={len(masks)} → using first {n}')

        img_lines  += [str(p) for p in imgs[:n]]
        mask_lines += [str(p) for p in masks[:n]]
        print(f'  {vid}: {n} pairs')

    out_img  = OUT_DIR / f'{split}.txt'
    out_mask = OUT_DIR / f'{split}2.txt'

    out_img.write_text('\n'.join(img_lines) + '\n')
    out_mask.write_text('\n'.join(mask_lines) + '\n')

    print(f'\nDone! Written:')
    print(f'  {out_img}  ({len(img_lines)} lines)')
    print(f'  {out_mask}  ({len(mask_lines)} lines)')


def main():
    all_videos = [d.name for d in sorted(IMG_DIR.iterdir()) if d.is_dir()]

    parser = argparse.ArgumentParser(description='Generate YOLO split txt files')
    parser.add_argument('--split', required=True,
                        choices=['train', 'val', 'test'],
                        help='Which split to generate (train/val/test)')
    parser.add_argument('--videos', nargs='*', default=[],
                        help='List of video folder names to include')
    parser.add_argument('--exclude', nargs='*', default=[],
                        help='Video folder names to EXCLUDE (used with --split train)')
    args = parser.parse_args()

    if args.videos:
        chosen = args.videos
    elif args.exclude:
        chosen = [v for v in all_videos if v not in args.exclude]
    else:
        # Default: all videos
        chosen = all_videos

    print(f'\nGenerating [{args.split}] split with {len(chosen)} video(s):')
    for v in chosen:
        print(f'  - {v}')
    print()

    gen_txt(chosen, args.split)


if __name__ == '__main__':
    main()
