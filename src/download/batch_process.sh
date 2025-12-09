mkdir -p data/raw/0-AsociacionCivil
mkdir -p data/raw/1-videolibros_private
mkdir -p data/raw/2-videolibros_public
mkdir -p data/raw/3-CNSordos
mkdir -p data/raw/4-Locufre

yt-dlp -f "bestvideo[height<=240][ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/" --write-subs --sub-langs "es.*"  https://www.youtube.com/@CNSORDOSARGENTINA -o "./data/raw/0-AsociacionCivil/0-%(video_autonumber)s.%(ext)s" --remote-components ejs:github
