#!/usr/bin/env python
# coding: utf-8

# ## Import packages

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from skimage.measure import regionprops, regionprops_table
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import load_img
from importlib import reload
import segmenteverygrain as seg
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from tqdm import trange
import matplotlib as qt

# ## Load models

# In[29]:


model = seg.Unet()
model.compile(optimizer=Adam(), loss=seg.weighted_crossentropy, metrics=["accuracy"])
model.load_weights('./segmenteverygrain/checkpoints/seg_model');

# the SAM model checkpoints can be downloaded from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")

# ## Run segmentation
# 
# Grains are supposed to be well defined in the image; e.g., if a grain consists of only a few pixels, it is unlikely to be detected.
# 
# The segmentation can take a few minutes even for medium-sized images, so do not start with large images (downsample them if necessary). Images with ~2000 pixels along their largest dimension are a good start.

# In[30]:


reload(seg)

#load images
# fname = '../images/bucegi_conglomerate_1_image.png'
# fname = '../images/A003_20201229_103823_image.png'
# fname = '../images/IMG_5208_image.png'
fname = '0a1d277fb473.tif'
# fname = '/Users/zoltan/Downloads/vecteezy_stone-pebbles-on-river-bed_3366528.jpg'
big_im = np.array(load_img(fname))

#Perform Segmentation
big_im_pred = seg.predict_big_image(big_im, model, I=256)
# decreasing the 'dbs_max_dist' parameter results in more SAM prompts (and longer processing times):
labels, grains, coords = seg.label_grains(big_im, big_im_pred, dbs_max_dist=10.0)
all_grains, labels, mask_all, grain_data, fig, ax = seg.sam_segmentation(sam, big_im, big_im_pred, coords, labels, min_area=50.0)

# Use this figure to check the distribution of SAM prompts (= black dots):

# In[27]:
dirname = 'output_directory/'
cv2.imwrite(dirname + fname.split('/')[-1][:-4] + '_mask.png', mask_all)

plt.figure()
plt.imshow(big_im_pred)
plt.scatter(coords[:,0], coords[:,1], c='k');

print(grain_data)

# ## Delete or merge grains in segmentation result
# * click on the grain that you want to remove and press the 'x' key
# * click on two grains that you want to merge and press the 'm' key (they have to be the last two grains you clicked on)

# In[547]:


grain_inds = []
cid1 = fig.canvas.mpl_connect('button_press_event', 
                              lambda event: seg.onclick2(event, all_grains, grain_inds, ax=ax))
cid2 = fig.canvas.mpl_connect('key_press_event', 
                              lambda event: seg.onpress2(event, all_grains, grain_inds, fig=fig, ax=ax))

# Run this cell if you do not want to delete / merge existing grains anymore; it is a good idea to do this before moving on to the next step.

# In[548]:


fig.canvas.mpl_disconnect(cid1)
fig.canvas.mpl_disconnect(cid2)

# Use this function to update the 'all_grains' list after deleting and merging grains:

# In[549]:


all_grains, labels, mask_all, fig, ax = seg.get_grains_from_patches(ax, big_im)

# Plot the updated set of grains:

# In[558]:


fig, ax = plt.subplots(figsize=(15,10))
ax.imshow(big_im)
plt.xticks([])
plt.yticks([])
seg.plot_image_w_colorful_grains(big_im, all_grains, ax, cmap='Paired')
seg.plot_grain_axes_and_centroids(all_grains, labels, ax, linewidth=1, markersize=10)
plt.xlim([0, np.shape(big_im)[1]])
plt.ylim([np.shape(big_im)[0], 0]);

# ## Add new grains using the Segment Anything Model
# 
# * click on unsegmented grain that you want to add
# * press the 'x' key if you want to delete the last grain you added
# * press the 'm' key if you want to merge the last two grains that you added
# * right click outside the grain (but inside the most recent mask) if you want to restrict the grain to a smaller mask - this adds a background prompt

# In[483]:


predictor = SamPredictor(sam)
predictor.set_image(big_im) # this can take a while
coords = []
cid3 = fig.canvas.mpl_connect('button_press_event', lambda event: seg.onclick(event, ax, coords, big_im, predictor))
cid4 = fig.canvas.mpl_connect('key_press_event', lambda event: seg.onpress(event, ax, fig))

# In[484]:


fig.canvas.mpl_disconnect(cid3)
fig.canvas.mpl_disconnect(cid4)

# After you are done with the deletion / addition of grain masks, run this cell to generate an updated set of grains:

# In[485]:


all_grains, labels, mask_all, fig, ax = seg.get_grains_from_patches(ax, big_im)

# ## Get grain size distribution

# Run this cell and then click (left mouse button) on one end of the scale bar in the image and click (right mouse button) on the other end of the scale bar:

# In[24]:


cid5 = fig.canvas.mpl_connect('button_press_event', lambda event: seg.click_for_scale(event, ax))

# Use the length of the scale bar in pixels (it should be printed above) to get the scale of the image (in units / pixel):

# In[27]:

n_of_units = 13.55 # centimeters in the case of 'IMG_5208_image.png'
units_per_pixel = n_of_units/492.06 # length of scale bar in pixels

# In[28]:


props = regionprops_table(labels.astype('int'), intensity_image = big_im, properties =\
        ('label', 'area', 'centroid', 'major_axis_length', 'minor_axis_length', 
         'orientation', 'perimeter', 'max_intensity', 'mean_intensity', 'min_intensity'))
grain_data = pd.DataFrame(props)
grain_data['major_axis_length'] = grain_data['major_axis_length'].values*units_per_pixel
grain_data['minor_axis_length'] = grain_data['minor_axis_length'].values*units_per_pixel
grain_data['perimeter'] = grain_data['perimeter'].values*units_per_pixel
grain_data['area'] = grain_data['area'].values*units_per_pixel**2

# In[29]:


grain_data.head()

print(grain_data.head)
# In[30]:


plt.figure()
plt.hist(grain_data['major_axis_length'], 25)
plt.xlabel('major axis length (cm)')
plt.ylabel('count');

# ## Save mask and grain labels to PNG files

# In[486]:


dirname = 'grayscale'
# write grayscale mask to PNG file
cv2.imwrite(dirname + fname.split('/')[-1][:-4] + '_mask.png', mask_all)
# Define a colormap using matplotlib
num_classes = len(all_grains)
cmap = plt.get_cmap('viridis', num_classes)
# Map each class label to a unique color using the colormap
vis_mask = cmap(labels.astype(np.uint16))[:,:,:3] * 255
vis_mask = vis_mask.astype(np.uint8)
# Save the mask as a PNG file
cv2.imwrite(dirname + fname.split('/')[-1][:-4] + '_labels.png', vis_mask)
# Save the image as a PNG file
cv2.imwrite(dirname + fname.split('/')[-1][:-4] + '_image.png', cv2.cvtColor(big_im, cv2.COLOR_BGR2RGB))

