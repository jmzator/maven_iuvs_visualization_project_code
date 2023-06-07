# This is a sample Python script.
import astropy
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
#image_filenames = [
#    'globe_sample_image_1.png',
#    'globe_sample_image_2.png',
#    'globe_sample_image_3.png',
#    'globe_sample_image_4.png',
#    'globe_sample_image_5.png',
#    'globe_sample_image_6.png',
#    'grid_image1.png',
#    'grid_image2.png',
#    'grid_image3.png',
#    'grid_image4.png',
#    'grid_image5.png',
#    'grid_image6.png',
#]

#for i, filename in enumerate(image_filenames):
#    image = Image.open(filename)
#    image = image.resize((tile_width, tile_height))
#    grid_image.paste(image, (i % num_cols * tile_width, i // num_cols * tile_height))

# save and show the grid image
#grid_image.save('grid_image.png')
# plt.show() not sure if can do that, just open the output file manually instead

### that worked for image grid of six sample png images!!!
### now try new method for making a gif of those sample images...

from PIL import Image
import glob

# path to the directory containing the png images
#image_folder = '/Users/jmzator/Desktop/maven_iuvs_visualization_project/'

# output gif filename
#output_file = '/Users/jmzator/Desktop/maven_iuvs_visualization_project/gif_outputs/clip_movie.gif'

# list all png files in the directory
#png_files = glob.glob(image_folder + "*.png")

# sort the files alphabetically (assuming they are named in order)
#png_files.sort()

# create a list to store the frames of the gif
#frames = []

# iterate over the png files and append each image as a frame
#for png_file in png_files:
#    img = Image.open(png_file)
#    frames.append(img)

# save the frames as an animated gif
#frames[0].save(output_file, format='GIF',
#               append_images=frames[1:],
#               save_all=True,
#               duration=2000,  # duration between frames in milliseconds
#               loop=0)  # 0 means an infinite loop, any other value specifies the number of loops

###### that worked for making a gif!
##### remember though that need to comment out the image grid first
#### when making gif, otherwise, code creates new grid image then
### adds that to gif...
### so maybe redo save path for that grid image....
### to change save path, maybe use os.path.join(save_path, file_name) ??


################
# now, here's code from Kyle email today 2023 May 22 regarding
# histogram equalization algorithm in python, a well-known
# data visualization algorithm and something needed for
# producing the globe images from IUVS apoapse data for this project
# several functions follow:
#
import numpy as np


def make_equidistant_spectral_cutoff_indices(n_spectral_bins: int) -> tuple[int, int]:
    """Make indices such that the input spectral bins are in 3 equally spaced color channels.

    Parameters
    ----------
    n_spectral_bins
        The number of spectral bins.

    Returns
    -------
    The blue-green and the green-red cutoff indices.

    Examples
    --------
    Get the wavelength cutoffs for some common apoapse MUV spectral binning schemes.

    >>> make_equidistant_spectral_cutoff_indices(15)
    (5, 10)

    >>> make_equidistant_spectral_cutoff_indices(19)
    (6, 13)

    >>> make_equidistant_spectral_cutoff_indices(20)
    (7, 13)

    """
    blue_green_cutoff = round(n_spectral_bins / 3)
    green_red_cutoff = round(n_spectral_bins * 2 / 3)
    return blue_green_cutoff, green_red_cutoff


def turn_detector_image_to_3_channels(image: np.ndarray) -> np.ndarray:
    """Turn a detector image into 3 channels by coadding over the spectral dimension.

    Parameters
    ----------
    image
        The image to turn into 3 channels. This is assumed to be 3 dimensional and have a shape of (n_integrations,
        n_spatial_bins, n_spectral_bins).

    Returns
    -------
    A co-added image with shape (n_integrations, n_spatial_bins, 3).

    """
    n_spectral_bins = image.shape[2]
    blue_green_cutoff, green_red_cutoff = make_equidistant_spectral_cutoff_indices(n_spectral_bins)

    red = np.sum(image[..., green_red_cutoff:], axis=-1)
    green = np.sum(image[..., blue_green_cutoff:green_red_cutoff], axis=-1)
    blue = np.sum(image[..., :blue_green_cutoff], axis=-1)

    return np.dstack([red, green, blue])


def histogram_equalize_grayscale_image(image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Histogram equalize a grayscale image.

    Parameters
    ----------
    image
        The image to histogram equalize. This is assumed to be 2-dimensional (2 spatial dimensions) but can have any
        dimensionality.
    mask
        A mask of booleans where :code:`False` values are excluded from the histogram equalization scaling. This must
        have the same shape as :code:`image`.

    Returns
    -------
    A histogram equalized array with the same shape as the inputs with values ranging from 0 to 255.

    See Also
    --------
    histogram_equalize_rgb_image: Histogram equalize a 3-color-channel image.

    Notes
    -----
    I could not get the scikit-learn algorithm to work so I created this.
    The algorithm works like this:

    1. Sort all data used in the coloring.
    2. Use these sorted values to determine the 256 left bin cutoffs.
    3. Linearly interpolate each value in the grid over 256 RGB values and the
       corresponding data values.
    4. Take the floor of the interpolated values since I'm using left cutoffs.

    """
    sorted_values = np.sort(image[mask], axis=None)
    left_cutoffs = np.array([sorted_values[int(i / 256 * len(sorted_values))]
                             for i in range(256)])
    rgb = np.linspace(0, 255, num=256)
    return np.floor(np.interp(image, left_cutoffs, rgb))


def histogram_equalize_rgb_image(image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Histogram equalize an RGB image.

    Parameters
    ----------
    image
        The image to histogram equalize. This is assumed to be 3-dimensional (the first 2 being spatial and the last
        being spectral). The last dimension as assumed to have a length of 3. Indices 0, 1, and 2 correspond to R, G,
        and B, respectively.
    mask
        A mask of booleans where :code:`False` values are excluded from the histogram equalization scaling. This must
        have the same shape as the first N-1 dimensions of :code:`image`.

    Returns
    -------
    A histogram equalized array with the same shape as the inputs with values ranging from 0 to 255.

    See Also
    --------
    histogram_equalize_grayscale_image: Histogram equalize a single-color-channel image.

    """
    red = histogram_equalize_grayscale_image(image[..., 0], mask=mask)
    green = histogram_equalize_grayscale_image(image[..., 1], mask=mask)
    blue = histogram_equalize_grayscale_image(image[..., 2], mask=mask)
    return np.dstack([red, green, blue])


def histogram_equalize_detector_image(image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """Histogram equalize a detector image.

    Parameters
    ----------
    image
        The image to histogram equalize. This is assumed to be 3-dimensional (the first 2 being spatial and the last
        being spectral).
    mask
        A mask of booleans where :code:`False` values are excluded from the histogram equalization scaling. This must
        have the same shape as the first N-1 dimensions of :code:`image`.

    Returns
    -------
    Histogram equalized IUVS image, where the output array has a shape of (M, N, 3).

    """
    coadded_image = turn_detector_image_to_3_channels(image)
    return histogram_equalize_rgb_image(coadded_image, mask=mask)

######### end Kyle's code from email

### now I'll try working with that code he provided

import astropy
from astropy.io import fits
a_file = astropy.io.fits.open('/Users/jmzator/Desktop/maven_iuvs_visualization_project/orbit18001/mvn_iuv_l1b_apoapse-orbit18001-muv_20230114T071804_v13_r01.fits')

#with fits.open(a_file) as hdul:
#    hdul.info()

fits.info('/Users/jmzator/Desktop/maven_iuvs_visualization_project/orbit18001/mvn_iuv_l1b_apoapse-orbit18001-muv_20230114T071804_v13_r01.fits')

make_equidistant_spectral_cutoff_indices(15)
#turn_detector_image_to_3_channels()
#histogram_equalize_grayscale_image()
#histogram_equalize_rgb_image()
#histogram_equalize_detector_image()

# Kyle convo 2023 May 23 lunch break from PSG: really only need to use
# the last function since that one calls the previous functions
# just read in the muv files don't need fuv, etc.
# the .gz on end of .fits is just compression and astropy knows how
# to uncompress when read in those files, so no need to manually unzip

# use numpy vstack, throw away all fuv, use only muv,
# stack those primary ones - prob dimensions like (200, x, 19) first one
#

# 2023 May 25 Thursday try quick create gif with smooth transitions

from PIL import Image, ImageSequence
import glob

# Path to the directory containing the PNG images
image_folder = '/Users/jmzator/Desktop/maven_iuvs_visualization_project/'

# Output GIF filename
output_file = '/Users/jmzator/Desktop/maven_iuvs_visualization_project/gif_outputs/clip_movie.gif'

# List all PNG files in the directory
png_files = glob.glob(image_folder + "*.png")

# Sort the files alphabetically (assuming they are named in order)
png_files.sort()

# Create a list to store the frames of the GIF
frames = []

# Iterate over the PNG files and append each image as a frame
for i in range(len(png_files)):
    img = Image.open(png_files[i])
    frames.append(img)

# Create a new list to store the frames with smooth transitions
transition_frames = []

# Iterate over the frames and apply cross-fade transitions
for i in range(len(frames)):
    # Fade-in transition for the first image
    if i == 0:
        transition_frames.append(frames[i])
    else:
        # Fade-in and fade-out transitions for the subsequent images
        duration = 500  # Duration of the cross-fade transition in milliseconds
        for alpha in range(0, 255, 5):  # Increase opacity from 0 to 255
            # Apply the cross-fade transition
            blended_frame = Image.blend(frames[i-1].convert("RGBA"), frames[i].convert("RGBA"), alpha/255.0)
            blended_frame = blended_frame.convert("RGB")
            transition_frames.append(blended_frame)
            transition_frames[-1].info["duration"] = duration
        transition_frames[-1].info["duration"] = duration

# Save the frames with smooth transitions as an animated GIF
transition_frames[0].save(output_file, format='GIF',
                          append_images=transition_frames[1:],
                          save_all=True,
                          optimize=False,
                          loop=0)  # 0 means an infinite loop, any other value specifies the number of loops

#### pretty sure failure is due to image sizes being different

###########

###########
# drop in the code from Kyle to Justin & Nick
# that gets ride of the seams in globe images
# will probably use something like this in future

# from Kyle email 2023-5-31 to Justin, Nick, and me
# "see the attached Python function, with documentation and comments
# that Zac added after I told him how it works. The idea is fairly
# straightforward: just convolve a 3x3 sharpening matrix with the
# image to be plotted. I skip the edges to avoid any potential problems
# but it should work over the remainder of the image"
# his code follows:

def sharpen_image(image):

"""

Take an image and sharpen it using a high-pass filter matrix:

|-----------|

| 0 -1 0 |

| -1 5 -1 |

| 0 -1 0 |

|-----------|



Parameters

----------

image : array-like

An (m,n,3) array of RGB tuples (the image).



Returns

-------

sharpened_image : ndarray

The original imaged sharpened by convolution with a high-pass filter.

"""



# the array I'll need to determine the sharpened image will need to be the size of the image + a 1 pixel border

sharpening_array = np.zeros((image.shape[0] + 2, image.shape[1] + 2, 3))



# fill the array: the interior is the same as the image, the sides are the same as the first/last row/column,

# the corners can be whatever (here they are just 0) (this is only necessary to sharpen the edges of the image)

sharpening_array[1:-1, 1:-1, :] = image

sharpening_array[0, 1:-1, :] = image[0, :, :]

sharpening_array[-1, 1:-1, :] = image[-1, :, :]

sharpening_array[1:-1, 0, :] = image[:, 0, :]

sharpening_array[1:-1, -1, :] = image[:, -1, :]



# make a copy of the image, which will be modified as it gets sharpened

sharpened_image = np.copy(image)



# multiply each pixel by the sharpening matrix

for integration in range(image.shape[0]):

for position in range(image.shape[1]):

for rgb in range(3):



# if the pixel is not a border pixel in sharpening_array, this will execute

try:

sharpened_image[integration, position, rgb] = \

5 * sharpening_array[integration + 1, position + 1, rgb] - \

sharpening_array[integration, position + 1, rgb] - \

sharpening_array[integration + 2, position + 1, rgb] - \

sharpening_array[integration + 1, position, rgb] - \

sharpening_array[integration + 1, position + 2, rgb]



# if the pixel is a border pixel, no sharpening necessary

except IndexError:

continue



# make sure new pixel rgb values aren't outside the range [0, 1]

sharpened_image = np.where(sharpened_image > 1, 1, sharpened_image)

sharpened_image = np.where(sharpened_image < 0, 0, sharpened_image)



# return the new sharpened image

return sharpened_image
