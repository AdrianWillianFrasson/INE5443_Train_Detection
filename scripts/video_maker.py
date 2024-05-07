import os

os.system('ffmpeg -i ../dataset/images/00/%03d.png -c:v libx264 "../dataset/videos/00.mp4"')
