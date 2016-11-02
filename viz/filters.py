'''Visualizing filters'''

import os
import sys
import time

import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt

def main():
	m = 2
	s = 1.
	k = 200
	x = np.linspace(-1,1,num=k)
	y = np.linspace(-1,1,num=k)
	X, Y = np.meshgrid(x, y)
	Theta = np.arctan2(Y, X)
	d = np.sqrt(x**2 + y**2)
	R = np.exp(-d**2/s)
	
	Wr = R*np.cos(m*Theta)
	Wi = R*np.sin(m*Theta)
	
	plt.figure(1)
	plt.imshow(Wr, interpolation='nearest', cmap='Blues')
	plt.axis('off')
	plt.figure(2)
	plt.imshow(Wi, interpolation='nearest', cmap='Reds')
	plt.axis('off')
	plt.show()

def gaussian(x,y):
    return np.exp(-(x**2 + y**2))

def plot_sinusoid():
	x = np.linspace(0,2*np.pi,num=360)
	plt.figure(1)
	plt.plot(np.cos(x), 'b', linewidth=5)
	plt.axis('off')
	plt.ylim([-1.05,1.05])
	
	plt.figure(2)
	plt.plot(np.sin(x), 'r', linewidth=5)
	plt.axis('off')
	plt.ylim([-1.05,1.05])
	
	plt.figure(3)
	plt.plot(np.cos(2*x), 'b', linewidth=5)
	plt.axis('off')
	plt.ylim([-1.05,1.05])
	
	plt.figure(4)
	plt.plot(np.sin(2*x), 'r', linewidth=5)
	plt.axis('off')
	plt.ylim([-1.05,1.05])
	plt.show()

def get_colors():
	for i in sns.color_palette("Blues_d"):
		print (255.*np.asarray(i)).astype(int)

if __name__ == '__main__':
	#main()
	#plot_sinusoid()
	get_colors()
























