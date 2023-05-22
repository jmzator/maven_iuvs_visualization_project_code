# This is a sample Python script.


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
#    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
#    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

###### start trying PyCharm more ####

import matplotlib.pyplot as plt

#whiteblankimage = 255 * np.ones(shape=[512, 512, 3], dtype=np.uint8)

#cv2.rectangle(whiteblankimage, pt1=(200,200), pt2=(300,300), color=(0,0,255), thickness=10)
#plt.imshow(whiteblankimage)
#plt.show()

#def makeGrid(nRows,nCols):
#   grid: list[list[Any]] = []
#   for i in range(nRows):
#     grid.append([])
#     for j in range(nCols):
#        grid[i].append('empty')
#
#        return grid

#print(makeGrid(3, 3))

#a = 5
#print('The value of a is', a)

### next tasks
### get first globe from any data (get code down)
### get data extraction process down (first real data set globe)
### make sure can export a grid of correct size for printing

#########
# 2023 May 17 try grid with image samples


#filename = 'globe_sample_image_1.png'

#imagegrid = plt.imread(filename)

#fig, axes = plt.subplots(2, 3)

#for row in [0, 1]:
#    for column in [0, 1, 2]:
#        ax = axes[row, column]
#        ax.set_title(f"Image ({row}, {column})")
#        ax.axis('off')
#        ax.imshow(imagegrid)

#plt.show()


### try create array
import cv2
import glob
import numpy as np


#X_data = []
#sample_pics = glob.glob("*.png")
#for myFile in sample_pics:
#    print(myFile)
#    image = cv2.imread(myFile)
#    X_data.append (image)

#print('X_data shape:', np.array(X_data).shape)
##error when print shape - inhomogenous




# image grid code example from image grid module:
#image-grid --folder ./images --n 4 --rows 1 --width 1000 --fill
#saved image-grid.jpg, (166, 998, 3), 27.1KiB


####################
### code below for making gif of images 2023 May 18
# adapted from Patrick O'Brien guest lecture ASTR3400

#from PIL import Image
#import glob
#frames = []
#imgs = sorted(glob.glob("Movie_{}M/*.png".format(int(resolution))))

#for i in imgs:
#    new_frame = Image.open(i)
#    frames.append(new_frame)

# Save into a GIF file that loops forever
#frames[0].save(‘out.gif’,
#        save_all = True, append_images = frames[1:], optimize = False, duration = duration, loop = 0)

### end gif code try
####################

##################
# 2023 May 22 Monday
# let's try new approach....

from PIL import Image

# define the dimensions of each image tile
tile_width = 100
tile_height = 100

# define the number of rows and columns in the grid
num_rows = 2
num_cols = 3

# create a new blank image for the grid
grid_width = tile_width * num_cols
grid_height = tile_height * num_rows
grid_image = Image.new('RGB', (grid_width, grid_height))

# load and resize the individual images
image_filenames = [
    'globe_sample_image_1.png',
    'globe_sample_image_2.png',
    'globe_sample_image_3.png',
    'globe_sample_image_4.png',
    'globe_sample_image_5.png',
    'globe_sample_image_6.png',
]

for i, filename in enumerate(image_filenames):
    image = Image.open(filename)
    image = image.resize((tile_width, tile_height))
    grid_image.paste(image, (i % num_cols * tile_width, i // num_cols * tile_height))

# save the grid image
grid_image.save('grid_image.png')
# plt.show() not sure if can do that, just open the output file instead

### that worked for image grid of six sample png images!!!
### now try new method for making a gif of those sample images...

from PIL import Image
import glob

# path to the directory containing the png images
image_folder = '/Users/jmzator/Desktop/maven_iuvs_visualization_project/'

# output gif filename
output_file = '/Users/jmzator/Desktop/maven_iuvs_visualization_project/gif_outputs/clip_movie.gif'

# list all png files in the directory
png_files = glob.glob(image_folder + "*.png")

# sort the files alphabetically (assuming they are named in order)
png_files.sort()

# create a list to store the frames of the gif
frames = []

# iterate over the png files and append each image as a frame
for png_file in png_files:
    img = Image.open(png_file)
    frames.append(img)

# save the frames as an animated gif
frames[0].save(output_file, format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=300,  # duration between frames in milliseconds
               loop=0)  # 0 means an infinite loop, any other value specifies the number of loops

###### that worked for making a gif!
##### remember though that need to comment out the image grid first
#### when making gif, otherwise, code creates new grid image then
### adds that to gif...
### so maybe redo save path for that grid image....
