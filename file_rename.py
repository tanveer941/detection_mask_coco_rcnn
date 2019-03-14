

import os
path = r'E:\Rujiao_ticket\Images'
files = os.listdir(path)

channel_name = 'MFC5xx_long_image_right'

for file in files:
    print(file)
    # os.rename(file, 'right_' + file)
    new_name = file.replace('screenshot_rgba_fourth.raw', 'rgbafourth')
    if 'screenshot_rgba_fourth.raw' in file:
        os.rename(os.path.join(path, file), os.path.join(path, new_name))