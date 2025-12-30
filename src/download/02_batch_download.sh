mkdir -p data/raw/0-AsociacionCivil
mkdir -p data/raw/1-videolibros_private
mkdir -p data/raw/2-videolibros_public
mkdir -p data/raw/3-CNSordos
mkdir -p data/raw/4-Locufre

yt-dlp \
-f "(bestvideo[vcodec^=avc][height<=1080]/bestvideo[vcodec^=avc])\
+bestaudio[ext=m4a]/best" \
--merge-output-format mp4 \
--write-subs \
--sub-langs "es.*" \
https://www.youtube.com/c/CanalesAsociaci%C3%B3nCivil/videos \
-o "./data/raw/0-AsociacionCivil/0-%(video_autonumber)s.%(ext)s"

yt-dlp \
-f "(bestvideo[vcodec^=avc][height<=1080]/bestvideo[vcodec^=avc])\
+bestaudio[ext=m4a]/best" \
--merge-output-format mp4 \
--write-subs \
--sub-langs "es.*" \
-a ./data/info/videolibros_private.txt \
-o "./data/raw/1-videolibros_private/1-%(video_autonumber)s.%(ext)s"

yt-dlp \
-f "(bestvideo[vcodec^=avc][height<=1080]/bestvideo[vcodec^=avc])\
+bestaudio[ext=m4a]/best" \
--merge-output-format mp4 \
--write-subs \
--sub-langs "es.*" \
https://www.youtube.com/@VideolibrosLSA/videos \
-o "./data/raw/2-videolibros_public/2-%(video_autonumber)s.%(ext)s"

yt-dlp \
-f "(bestvideo[vcodec^=avc][height<=1080]/bestvideo[vcodec^=avc])\
+bestaudio[ext=m4a]/best" \
--merge-output-format mp4 \
--write-subs \
--sub-langs "es.*" \
https://www.youtube.com/@CNSORDOSARGENTINA/videos \
-o "./data/raw/3-CNSordos/3-%(video_autonumber)s.%(ext)s"

yt-dlp \
-f "(bestvideo[vcodec^=avc][height<=1080]/bestvideo[vcodec^=avc])\
+bestaudio[ext=m4a]/best" \
--merge-output-format mp4 \
--write-subs \
--sub-langs "es.*" \
https://www.youtube.com/channel/UCPJr7e9V_07DAID60F0pXVw \
-o "./data/raw/4-Locufre/4-%(video_autonumber)s.%(ext)s"