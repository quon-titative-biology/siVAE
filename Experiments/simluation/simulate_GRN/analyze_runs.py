# Analyze result from different runs
## Plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import offsetbox
from matplotlib import cm
import seaborn as sns

import os
import numpy as np

import argparse
parser          = argparse.ArgumentParser()
logdir          = parser.add_argument('--logdir')
args = parser.parse_args()
# Directories
logdir          = args.logdir
root            = os.path.join(logdir,'').rsplit('/',2)[0]


## Combine and plot the losses
list_randomseed = [x for x in os.listdir(root) if os.path.isdir(os.path.join(root,x))]

curr_losses     = np.array([np.loadtxt(os.path.join(root, rseed, 'losses.csv'), delimiter=',', skiprows = 1) for rseed in list_randomseed])
sorted_idx      = np.argsort(curr_losses,0)[:,0]
curr_losses     = curr_losses[sorted_idx]
list_randomseed = [list_randomseed[ii] for ii in sorted_idx]

with open(os.path.join(root, list_randomseed[0], 'losses.csv')) as f:
    header = f.readline().replace('\n','')

# Save as csv
np.savetxt(os.path.join(root,'losses.csv'),
           curr_losses,
           delimiter = ',',
           header    = header,
           comments   = '')

# Plot
dict_error = {'Recon_loss': curr_losses[:,1],
              'Total_loss': curr_losses[:,0]}

fig = plt.figure()
index = np.arange(curr_losses.shape[0])
bar_width = 1/(len(dict_error)+1.)
opacity = 0.8
i = -1
seed_index = [x.rsplit('-',1)[1] for x in list_randomseed]

for key,value in dict_error.iteritems():
    i = i+1
    print(i)
    plot = plt.bar(index + bar_width * i, np.abs(value), bar_width, alpha = opacity, label = key)
    print(key)
    plt.xlabel('Random Seed')
    plt.ylabel('')
    plt.title('Error')
    plt.xticks(index, seed_index)
    plt.legend()
    plt.tight_layout()
plt.savefig(os.path.join(root, 'Losses'))
plt.close()

## Combine Images
import sys
from PIL import Image

im_name_list = ['Heatmaps_sample', 'heatmap_DE_methods', 'recon_error', 'heatmap_DE_methods_0-2', 'heatmap_DE_methods_decoder', 'recon_baseline']
im_list = [map(Image.open, [os.path.join(root,x,im_name+'.png') for x in list_randomseed]) for im_name in im_name_list]
widths, heights = zip(*(i.size for i in im_list[0]))
width_new = np.sum(widths)
height_new = np.max(heights) * len(im_name_list)
new_im = Image.new('RGB', (width_new, height_new))

y_offset = 0
for ims in im_list:
    x_offset = 0
    for im in ims:
        print y_offset
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
    y_offset += im.size[1]

new_im.save(os.path.join(root, 'combined_images.png'))
