# Created by Chang Liu, Boston University
# cliu17@bu.edu
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt


# Data preperation. Generate random long-tail data and save it
mu, sigma = 5., 0.7   # mean and standard deviation
s = np.random.lognormal(mu, sigma, 1500) 
# Comment following code if you have no intention to keep a stable figure
np.savetxt('original.csv', s, delimiter=',')    # save data
original = np.loadtxt(open("original.csv", "r"), delimiter=",")    # load data

# Set figure size and distance between subplots
plt.figure(figsize=(16,5), dpi =100)  
plt.subplots_adjust(hspace=0.2, wspace=0.3) 

# Create subplot 1
plt1 = plt.subplot(131)
plt1.set_title('Original Histogram', fontsize=20, pad=25)  # Set title
plt1.tick_params(labelsize=20)
plt1.set_xticks((0, 500, 1000, 1500))  # set x-axis label
plt.hist(original, 100, density=False, align='mid', color='cornflowerblue') 

# Create subplot2
plt2 = plt.subplot(132)
log = np.log(original)  # log transformation
plt2.set_title('Log Transformed Histogram',fontsize=20, pad=25)  # Set title
plt2.set_xlim((3, 8))  # set x-axis and y-axis limit to keep two transformation results in the same scope
plt2.set_ylim((0, 60))
plt2.tick_params(labelsize=20)
plt2.set_xticks((3, 5, 7)) # set x-axis label
plt.hist(log, 100, color='cornflowerblue')


# Create subplot3
plt3 = plt.subplot(133)
boxcox = stats.boxcox(original)[0]  # Box-cox log transformation
plt3.set_title('Box-Cox Transformed Histogram', fontsize=20, pad=25)  # Set title
plt3.set_xlim((3, 8))  # set x-axis and y-axis limit to keep two transformation results in the same scope
plt3.set_ylim((0, 60))
plt3.set_xticks((3, 5, 7)) # set x-axis label
plt3.tick_params(labelsize=20)
plt.hist(boxcox, 100, color='cornflowerblue')


plt.show()
