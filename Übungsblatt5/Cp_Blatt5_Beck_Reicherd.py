# kein kompilieren notwendig
# python3 CP_Blatt5_Beck_Reicherd.py
# Mit dem obigen Befehl kann das Programm ausgeführt werden. Es werden dann alle Graphen sowohl angezeigt als auch abgespeichert.
# K. Beck, M. Reicherd
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
def fft(f):
    n= len(f)

    if n == 1:
        return f
    else:
        f_even = fft(f[::2])
        f_odd = fft(f[1::2])
        a = np.exp(-2j*np.pi*np.arange(n)/ n)

        c = np.concatenate([f_even+a[:int(n/2)]*f_odd,f_even+a[int(n/2):]*f_odd])
        return c



n=512
x= np.linspace(0,10,n)
y= 6*np.sin(45*2*np.pi*x)+ 1*np.cos(20*2*np.pi*x)+ 2*np.sin(5*2*np.pi*x)
c = np.empty(int(n))
a=np.abs(fft(y))/(n/2)
x1= np.linspace(0,n/10,n)
i= np.where(a==np.max(a))
print(i)
plt.plot(x1,a)
plt.show()
plt.plot(x,y)
plt.show()
"""




arr = []
exponents = []
# Funktion, um die Bedingung len(array) = 2^n für FFT zu erfüllen
def fillWithZeros(a):
    N = len(a)
    exponent = np.log2(N)
    exponent= int(exponent+1)
    newN = 2**exponent
    if (newN-N)%2!=0:
        return np.concatenate([np.zeros(int((newN - N) / 2)+1), a, np.zeros(int((newN - N) / 2))])

    return np.concatenate([np.zeros(int((newN-N)/2)),a,np.zeros(int((newN-N)/2))])


# Fast Fourier Transformation - Vortransformation (Signal -> Frequenz)
def FFT(f):
    N = len(f)
    if N <= 1:
        return f

    # Einteilung in gerade und ungerade Indizees
    even = FFT(f[0::2])
    odd = FFT(f[1::2])

    # Array in dem Ergebnisse gespeichert werden
    temp = np.zeros(N).astype(np.complex64)

    # Iteration der Hälfte der Daten
    for u in range(N // 2):
        temp[u] = even[u] + np.exp(-2j * np.pi * u / N) * odd[u]  # conquer
        temp[u + N // 2] = even[u] - np.exp(-2j * np.pi * u / N) * odd[u]  # conquer

    return temp


# Fast Fourier Transformation - Rücktransformation (Frequenz -> Signal)
def iFFT(f):
    N = len(f)
    if N <= 1:
        return f

    # division
    even = FFT(f[0::2])
    odd = FFT(f[1::2])

    # store combination of results
    temp = np.zeros(N).astype(np.complex64)

    # only required to compute for half the frequencies
    # since u+N/2 can be obtained from the symmetry property
    for u in range(N // 2):
        temp[u] = 1 / N *even[u] + 1 / N * np.exp(2j * np.pi * u / N) * odd[u]  # conquer
        temp[u + N // 2] = 1 / N * even[u] - 1 / N * np.exp(2j * np.pi * u / N) * odd[u]  # conquer

    return temp

Data = pd.read_csv("GW150914NRdat.txt", sep=" ", header=0)
time = Data.loc[:, "#"].to_numpy()
signal = Data.loc[:, "time"].to_numpy()

def FFTTest():
    x= np.linspace(0,20,1024)
    y=1*np.sin(2*np.pi*x)+2*np.sin(3*2*np.pi*x)+3*np.sin(2*2*np.pi*x)
    y1=FFT(y)
    plt.plot(np.linspace(0,1024/20,1024)[:100],np.abs(y1)[:100])
    plt.show()
    y2= iFFT(y1)
    plt.plot(x,y)
    plt.plot(x,np.multiply(y2,-1))
    plt.show()

def masseBestimmen():
    timeIndeces = np.where((time>0.39) & (time<0.425))

    plt.plot(time[timeIndeces],signal[timeIndeces])
    plt.show()
    t1=time[timeIndeces]
    print(len(t1))
    diff=[]
    for i in range(0,len(t1)-1):
      diff.append(t1[i+1]-t1[i])
    samplingFrequence=1/np.average(diff)
    frequences=[]
    for t in timeIndeces[0]:

        intervalSignal = signal[range(t,t+512,1)]

        fftSignal = np.abs(FFT(intervalSignal))
        frequences.append(fftSignal.argmax()*samplingFrequence/512)
        #plt.plot(np.arange(0,128,1),fftSignal)
        #plt.plot(time[range(t,t+256,1)],intervalSignal)
        #plt.show()
    a = np.polyfit(t1, np.power(frequences, -8 / 3), 1)
    print(a)
    plt.plot(t1,t1*a[0]+a[1])
    plt.plot(t1,np.power(frequences,-8/3))
    plt.show()
    a=np.polyfit(t1,np.power(frequences,-8/3),1)

    #intervalSignal=fillWithZeros(signal[timeIndeces])
    #fftSignal= FFT(intervalSignal)
    #plt.plot(time[timeIndeces],signal[timeIndeces])
    #plt.show()
    #plt.plot(np.linspace(0,samplingFrequence,len(t1))[:100],np.abs(fftSignal)[:100])
    #plt.show()

masseBestimmen()

"""
plt.plot(time,signal)
plt.show()
#time = fillWithZeros(time)
signal = fillWithZeros(signal)
transformedSignal=FFT(signal)
plt.plot(np.arange(0,2**12,1)[0:150],np.abs(transformedSignal[0:150]))
plt.show()
"""

"""
deltaN = (len(time)-len(divider(signal)[0][0]))-1
transformedvals = np.zeros(int(deltaN/2), dtype = int)
transformedvals = np.concatenate((transformedvals,FFT(divider(signal)[0][0]),transformedvals))
intensity = np.absolute(transformedvals)**2
maxintensity = intensity[1:].argmax()+1  # returns indices of the max element of the array
time = time[:-1]
print(maxintensity)
plt.loglog(2*np.pi/time,intensity)
plt.plot(2*np.pi/time[maxintensity],intensity[maxintensity],'*',c='red',markersize= 10)
# plt.xlim(2*np.pi/.39,2*np.pi/.425)
print(2*np.pi/time[maxintensity])
plt.show()

plt.plot(time,2*np.pi/time,'*',c='red',markersize= 10)
plt.xlabel(r'time [s]')
plt.ylabel(r'Frequenz')
plt.xlim(.39,.425)
plt.show()

# print(len(transformedvals))
# print(len(time))

"""
