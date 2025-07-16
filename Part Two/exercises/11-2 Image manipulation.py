import matplotlib
from matplotlib import image
from matplotlib import pyplot
import os

import numpy as np

# Read an image file
path = os.path.dirname(os.path.abspath(__file__))
filename = path + '/' + 'lenna.bmp'
data = image.imread(filename)

# Display image information
print('Image type is: ', type(data))
print('Image shape is: ', data.shape)

path2 = os.path.dirname(os.path.abspath(__file__))
filename2 = path + '/' + 'flag.png'
data2 = image.imread(filename2)

# height = data2.shape[0]
# print('Image height is: ', height)
# width = data2.shape[1]
# print('Image width is: ', width)

# Add some color boundaries to modify an image array
plot_data = data.copy()
plot_data = plot_data.astype(np.float32) / 255.0  # Ensure the data type is correct for image saving

for width in range(0, 250):
    for height in range(250):

        color = data2[height, width]
        print("plot color:", color[0], color[1], color[2])
        plot_data[height, width+262] = [color[0], color[1], color[2]]  #why is the data2 not being put in plotdata?
        # print("plot data after:", plot_data[height][width])

        
# Write the modified images
image.imsave(path+'/'+'lenna-mod.jpg', plot_data)

# use pyplot to plot the image
pyplot.imshow(plot_data)
pyplot.show()