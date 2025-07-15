ffmpeg -i buffer.mp4 \
       -map 0:v:0 -map 0:a:0 \
       -c:v libx264 -b:v 1500k \
       -c:a aac -b:a 128k \
       -f dash \
       -seg_duration 4 \
       -use_template 1 \
       -use_timeline 1 \
       -init_seg_name 'init-stream$RepresentationID$.m4s' \
       -media_seg_name 'chunk-stream$RepresentationID$-$Number%05d$.m4s' \
       dash_output/tv002/tv002.mpd


uv run video_to_subtitle.py buffer.mp4 dash_output/tv002/tv002.vtt --language zh 
