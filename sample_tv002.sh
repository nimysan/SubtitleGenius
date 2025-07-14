ffmpeg -i arabic-long.webm \
  -map 0:v -c:v libx264 -b:v:0 1500k -s:v:0 1280x720 \
  -map 0:v -c:v libx264 -b:v:1 800k -s:v:1 854x480 \
  -map 0:a -c:a aac -b:a 128k \
  -f dash -use_template 1 -use_timeline 1 -window_size 4 \
  -adaptation_sets "id=0,streams=v id=1,streams=a" \
  dash_output/tv002/tv002.mpd
