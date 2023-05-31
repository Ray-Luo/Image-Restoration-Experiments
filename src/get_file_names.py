import os
import pandas as pd

# Specify the directory you want to scan
folder_path = '/home/luoleyouluole/Image-Restoration-Experiments/data/HDR_VIDEO_FRAME_4xd_patchify'

# Get a list of all files in the directory
file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Create a pandas DataFrame from the list
df = pd.DataFrame(file_list, columns=['filename'])

# Write the DataFrame to a CSV file
df.to_csv('HDR_VIDEO_FRAME_patch_lq.csv', index=False)
