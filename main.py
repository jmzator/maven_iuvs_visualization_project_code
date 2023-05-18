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


filename = 'globe_sample_image_1.png'

imagegrid = plt.imread(filename)

fig, axes = plt.subplots(2, 3)

for row in [0, 1]:
    for column in [0, 1, 2]:
        ax = axes[row, column]
        ax.set_title(f"Image ({row}, {column})")
        ax.axis('off')
        ax.imshow(imagegrid)

plt.show()


### try create array
import cv2
import glob
import numpy as np

X_data = []
sample_pics = glob.glob("*.png")
for myFile in sample_pics:
    print(myFile)
    image = cv2.imread(myFile)
    X_data.append (image)

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
#save_all = True, append_images = frames[1:], optimize = False, duration = duration, loop = 0)

### end gif code try
####################