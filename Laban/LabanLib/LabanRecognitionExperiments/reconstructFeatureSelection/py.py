#x= [1,5,10,20,30,40,50,60,80]
#y = [0.005,0.02,0.038,0.024,0.021,0.015,0.01,0.005,0.001]
x=[0.3,0.5,0.6,0.7,0.8,0.9,1]
y=[0.556,0.563,0.569,0.564,0.565,0.5645,0.556]
import matplotlib.pylab as plt
import matplotlib
font = {'family' : 'normal',
        'style' : 'italic',
        'size'   : 20}
matplotlib.rc('font', **font)
plt.plot(x,y)
#plt.xlabel('Percent of features that were selected ')
plt.xlabel('Fraction of selected features')
#plt.ylabel('Improvement in F1 score')
plt.ylabel('F1 score')
plt.grid()
plt.show()