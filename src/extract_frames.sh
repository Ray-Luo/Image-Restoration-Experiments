#!/bin/bash

# Directory where the videos are stored
video_dir="/home/luoleyouluole/Image-Restoration-Experiments/data/LIVE_HDR_Public/videos"

# File name prefix
file_prefix="4k_ref_"

# Directory where you want to save frames
output_dir="/home/luoleyouluole/Image-Restoration-Experiments/data/LIVE_HDR_Public/frames"

# Loop over all mp4 files in the directory
for file in "$video_dir"/"$file_prefix"*.mp4
do
  # Get base name of the file (without directory)
  base_name=$(basename "$file")

  # Remove .mp4 extension from the base name
  base_name_without_ext="${base_name%.*}"

  # Use ffmpeg to extract frames
  ffmpeg -i "$file" -vf "select=not(mod(n\,60))" -vsync vfr -pix_fmt rgb48be "$output_dir"/"$base_name_without_ext"_%04d.png
done
