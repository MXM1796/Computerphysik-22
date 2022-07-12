# kein kompilieren notwendig
# python3 CP_Blatt5_Beck_Reicherd.py
# Mit dem obigen Befehl kann das Programm ausgeführt werden. Es werden dann alle Graphen sowohl angezeigt als auch abgespeichert.
# K. Beck, M. Reicherd
import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Funktion, um die Bedingung len(array) = 2^n für FFT zu erfüllen
def fillWithZeros(a):
    N = len(a)
    exponent = np.log2(N)
    exponent= int(exponent+1) #rundet Exponenten auf nächst größere Ziffer auf
    newN = 2**exponent
    if (newN-N)%2!=0:
        return np.concatenate([np.zeros(int((newN - N) / 2)+1), a, np.zeros(int((newN - N) / 2))]) #symmetrische Hinzufügung von Nullen in den Array
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

    # Iteration über N/2 durch Symmetrie in komplexen Bereich
    for u in range(N // 2):
        temp[u] = even[u] + np.exp(-2j * np.pi * u / N) * odd[u]
        temp[u + N // 2] = even[u] - np.exp(-2j * np.pi * u / N) * odd[u]

    return temp


# Fast Fourier Transformation - Rücktransformation (Frequenz -> Signal)
def iFFT(f):
    N = len(f)
    if N <= 1:
        return f

    # Einteilung in gerade und ungerade Indizees
    even = iFFT(f[0::2])
    odd = iFFT(f[1::2])

    # Array in dem Ergebnisse gespeichert werden
    temp = np.zeros(N).astype(np.complex64)

    # Iteration über N/2 durch Symmetrie in komplexen Bereich
    for u in range(N // 2):
        temp[u] = 1 / N * even[u] + 1 / N * np.exp(2j * np.pi * u / N) * odd[u]
        temp[u + N // 2] = 1 / N * even[u] - 1 / N * np.exp(2j * np.pi * u / N) * odd[u]

    return temp

Data = pd.read_csv("GW150914NRdat.txt", sep=" ", header=0)
time = Data.loc[:, "#"].to_numpy()
signal = Data.loc[:, "time"].to_numpy()


def FFTTest():
    diff =[]

    for i in range(0,len(time)-1):
      diff.append(time[i+1]-time[i])
    samplingFrequence=1/np.average(diff) #Frequenzresolution

    Signal = fillWithZeros(signal)
    fftSignal= FFT(Signal)
    plt.plot(np.linspace(0,samplingFrequence,4096)[:50],np.abs(fftSignal)[:50]) #Vortransformation
    plt.ylabel(r"Relative Intensität")
    plt.xlabel(r"$(Frequenz)$ [1/u]")
    plt.savefig("FFT_Übung5")
    plt.show()

    y2 = iFFT(fftSignal)
    plt.plot(time[:len(time)],y2[:len(time)]) #Rücktranstransformation
    plt.ylabel(r"Amplitude")
    plt.xlabel(r"$Zeit$ [u]")
    plt.savefig("iFFT_Übung5")
    plt.show()



FFTTest()


def frequenz_zeit():
    timeIndeces = np.where((time>0.39) & (time<0.425))
    t1=time[timeIndeces]#Array mit allen Zeiten aus dem gesuchten Intervall
    diff=[]

    for i in range(0,len(t1)-1):
      diff.append(t1[i+1]-t1[i])
    samplingFrequence=1/np.average(diff) #Samplingfrequenz für die Frequenzresolution

    frequences=[] #Array für Frequenzen

    for t in timeIndeces[0]:

        intervalSignal = signal[range(t,t+512,1)]#Bestimmung der Signalintervalle
        fftSignal = np.abs(FFT(intervalSignal)) #Bestimmung der FFT der Signalintervalle
        frequences.append(fftSignal.argmax()*samplingFrequence/512)

    plt.plot(t1,frequences) #Auftragung von Frequenzen gegen Zeit
    plt.xlabel(r"Zeit [u]")
    plt.ylabel(r"$Frequenz$ [1/u]")
    plt.savefig('f_GW_t')
    plt.show()

    a = np.polyfit(t1, np.power(frequences, -8 / 3), 1) #Angepasster Geradenfit der potenzierten Frequenzen
    print('Die Steigung beträgt: ',a[0], '. Der Y-Achsenabschnitt ist: ',a[1])
    plt.plot(t1,t1*a[0]+a[1]) #
    plt.plot(t1,np.power(frequences,-8/3))
    plt.xlabel(r"Zeit [u]")
    plt.ylabel(r"$Frequenz^{-8/3}$ [1/u]")
    plt.savefig('Geradenfit_FFt')
    plt.show()

    intervalSignal=fillWithZeros(signal[timeIndeces]) # Fouriertransformation für die Zeitintervalle
    fftSignal= FFT(intervalSignal)
    plt.plot(np.linspace(0,samplingFrequence,len(t1))[:100],np.abs(fftSignal)[:100])
    maxfreq  = np.abs(fftSignal).argmax()
    maxint = np.linspace(0,samplingFrequence,len(t1))[maxfreq]
    plt.ylabel(r"Relative Intensität")
    plt.xlabel(r"$(Frequenz)$ [u]")
    plt.savefig('FFT_Zeitintervalle')
    plt.show()

    return maxint


c = 299792458
G = 6.67430 * 10 ** (-11)
Ms = 1.98892 * 10 ** 30
f = frequenz_zeit()
m = 0.000292543

def massebestimmen():
    chirpmasse = ((m * 5/(8*np.pi)**(8/3))**(3/5) * (c**3)/G)
    masseges = ((chirpmasse**5 * 8)**(1/5))
    r = (G*masseges/(np.pi*f)**2)**(1/3) * 10**(-3)
    rs = 2*G*masseges/c**2 * 10**(-3)
    return chirpmasse,masseges,r,rs

print(r"Die Chirpasse beträgt: ", massebestimmen()[0]/Ms,"Sonnenmassen")
print(r"Die Gesamtmasse beträgt: ", massebestimmen()[1]/Ms,"Sonnenmassen")
print(r"Der Abstand beträgt: ", massebestimmen()[2],"[Km]")
print(r"Der Schwarzschildradius beträgt: ", massebestimmen()[3],"[Km]")
