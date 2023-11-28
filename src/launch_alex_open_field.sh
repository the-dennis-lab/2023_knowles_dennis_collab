#!/bin/env bash
#conda activate ejd_fieldwork

# change these paths to the path with all of your videos
declare -a LIST_OF_FILES=("/home/dennislab2/Desktop/videos/repeat_0pt8_threshold/*.csv")
echo "$LIST_OF_FILES"

for n in $LIST_OF_FILES
do
  echo "$n"
  python alex_open_field.py $n '../data/data/provided/20230314_kmeans_jumps_rears.pkl' '../data/provided_data/start_times_videos_april_2023.csv'
done

cat ../data/results/*_summary.csv > april23_repeats_combined.csv
