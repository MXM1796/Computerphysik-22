#kein kompilieren notwendig
#python3 CP_Blatt2_Beck_Reicherd.py
#Mit dem obigen Befehl kann das Programm ausgeführt werden. Es werden dann die beiden Graphen sowohla angezeigt als auch abgespeichert.
# K. Beck, M. Reicherd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

a = pd.read_csv("abweichungdat.sec", sep=" ", encoding='ISO-8859-1')# Einlesen der  Werte
y1 = a._get_column_array(1) #Arrays initioalieseren
x1 = a._get_column_array(0)
plt.scatter(x1,y1)
plt.xlabel("h")# x-Achse beschriften
plt.ylabel(r"$\Delta_g$")# y-Achse beschriften
plt.savefig("B1_Graphik1")# Graph speichern
plt.show()#Graph plotten

x_v=[]
y_v=[] #Arrays für Verfahrensfehler

x_m=[]
y_m=[] #Arrays für Maschinenfehler

for i in range(0,y1.size):
    if x1[i]>0.001: #Alle Werte die >0.001 sind, werden zur Verfahrensfehlerseite hinzugefügt
        x_v.append(x1[i])
        y_v.append(y1[i])
    else: #die restlichen zur Maschinenfehlerseite
        x_m.append(x1[i])
        y_m.append(y1[i])

##################### erster Geradenfit ###################
x_vLog=np.log(x_v) #Logarithmieren der Skalen
y_vLog=np.log(y_v)
average_x= np.average(x_vLog) # Berechnung verschiedener Mittelwerte
average_y= np.average(y_vLog)
average_x2= np.average(np.power(x_vLog,2))
average_xy= np.average(np.multiply(x_vLog,y_vLog))
m1= (average_xy-average_x*average_y)/(average_x2-average_x**2)# Berechnung der Geraden
n1= (average_x2*average_y-average_x*average_xy)/(average_x2-average_x**2)
chi= np.average(np.power(np.add(y_vLog,np.add(np.multiply(-m1,x_vLog),-n1)),2))# Berechnung des mittleren Abstandsquadrats
print("m1="+str(m1)+",  n1="+str(n1),"durchschnittliches Abstandsquadrat:"+str(chi))
x_fit=np.arange(np.max(-12.0),np.max(x_vLog),0.01)# Berechnen der Geradenpunkte
y_fit=x_fit*m1+n1
plt.figure(figsize=(10, 10))
plt.xlabel(r"$\ln(h)$")# Beschriften
plt.ylabel(r"$\ln(\Delta_g)$")
plt.plot(x_fit,y_fit,label="Geradenfit v")#plotten
plt.scatter(x_vLog,y_vLog,4,color='red',label="Datenpunkte")
plt.legend()



##################### zweiter Geradenfit ##################
x_mLog=np.log(x_m) #Logarithmieren der Skalen
y_mLog=np.log(y_m)
average_x= np.average(x_mLog) # Berechnung verschiedener Mittelwerte
average_y= np.average(y_mLog)
average_x2= np.average(np.power(x_mLog,2))
average_xy= np.average(np.multiply(x_mLog,y_mLog))
m2= (average_xy-average_x*average_y)/(average_x2-average_x**2)# Berchnung der Geraden
n2= (average_x2*average_y-average_x*average_xy)/(average_x2-average_x**2)
chi= np.average(np.power(np.add(y_mLog,np.add(np.multiply(-m2,x_mLog),-n2)),2))# Berechnung des mittleren Abstandsquadrats
print("m2="+str(m2)+",  n2="+str(n2),"durchschnittliches Abstandsquadrat:"+str(chi))
x_fit=np.arange(np.min(x_mLog),-5.0,0.01)# Berechnen der Geradenpunkte
y_fit=x_fit*m2+n2
plt.xlabel(r"$\ln(h)$")# Beschriften
plt.ylabel(r"$\ln(\Delta_g)$")
plt.plot(x_fit,y_fit,label="Geradenfit m")#plotten
plt.scatter(x_mLog,y_mLog,4,color='red')
plt.legend()

plt.savefig("Geradenfits")# Graphik speichern
plt.show()# Anzeigen der Graphik


##################### Aufgabenteil zwei ###################
ZweiPi = 2 * np.pi
w_arr = np.arange(1, 21)
h = np.linspace(0.0001, 0.3, 1000)#Initialisieren von h



# Zwei Punkt Formel
def d_y_approx_2(w, h, x0):
    d_y_a = np.divide(np.cos(ZweiPi * w *(x0/w +h)) - np.cos(ZweiPi * w *(x0/w)),h)
    return d_y_a

# exakte Ableitung
def d_y_exakt(w,x0):
    d_y_e = - (ZweiPi * w) * np.sin(ZweiPi * w * (x0 / w))
    return d_y_e

# drei Punkt Formel
def d_y_approx_3(w, h,x0):
    d_y_a_3 = np.divide(np.cos(ZweiPi * w *(x0/w +h)) - np.cos(ZweiPi * w *(x0/w -h)),2*h)
    return d_y_a_3


def get_h_max(x,y1,y2,maxDiff):# Die Funktion vergleich zwei Arrays y1 und y2
    for i in range(0,x.size):  # und gibt den x-wert zurück, bei dem y1 -y2 das
        if np.abs(y1[i]-y2[i])>maxDiff:#erste mal stärker als maxDiff von y2 abweicht.
            return np.exp(x[i])

def get_h_max_array_2point(h,x,maxDiff,x0): #Berechnet h_max(\omega) für 2 punkt
    a=[]
    for w in np.linspace(1,20,10000):
        y_values=np.log(np.abs(np.add(d_y_approx_2(w,h,x0),-d_y_exakt(w,x0))))#Berechnung von delta_g
        y_expected=x-x[0]+y_values[0]#Berechnung der erwarteten Gerade
        a.append(get_h_max(x,y_values,y_expected,maxDiff))

    return a
def get_h_max_array_3point(h,x,maxDiff,x0):#Berechnet h_max(\omega) für 3 punkt
    a=[]
    for w in np.linspace(1,20,10000):
        y_values=np.log(np.abs(np.add(d_y_approx_3(w,h,x0),-d_y_exakt(w,x0))))#Berechnung von delta_g
        y_expected=x*2-x[0]*2+y_values[0]#Berechnung der erwarteten Gerade
        a.append(get_h_max(x,y_values,y_expected,maxDiff))

    return a


#Plots für x_1
x0=0.1
x_values=np.log(h)
y_values_1=np.log(np.abs(np.add(d_y_approx_2(5,h,x0),-d_y_exakt(5,x0))))#delta_g berechnen für omega=2

y_values_2=np.log(np.abs(np.add(d_y_approx_3(5,h,x0),-d_y_exakt(5,x0))))#delta_g berechnen für omega=2

y_expected1=x_values-x_values[0]+y_values_1[0]#Berechnung der erwarteten Geraden
y_expected2=x_values*2-x_values[0]*2+y_values_2[0]
plt.plot(x_values,y_expected1,label="erwartet 2punkt")
plt.plot(x_values,y_expected2,label="erwartet 3punkt")
plt.plot(x_values,y_values_1,label="2punkt")
plt.plot(x_values,y_values_2,label="3punkt")
plt.legend()
plt.xlabel(r"$\ln{h}$")
plt.ylabel(r"$\ln{\Delta_g}$")
plt.savefig("Verfahrensfehler_x1");
plt.show()
print("hmax für x1,2punkt und omega=5: "+str(get_h_max(x_values,y_values_1,y_expected1,1)))
print("hmax für x1,3punkt und omega=5: "+str(get_h_max(x_values,y_values_2,y_expected2,1)))
plt.xlabel(r"$\omega$")
plt.ylabel(r"$h_{max}$")
plt.plot(np.linspace(1,21,10000),get_h_max_array_2point(h,x_values,1,x0),label="2punkt")
plt.plot(np.linspace(1,21,10000),get_h_max_array_3point(h,x_values,1,x0),label="3punkt")
plt.legend();
plt.savefig("Hmax_von_omega_1")
plt.show()



x0=0.249
x_values=np.log(h)
y_values_1=np.log(np.abs(np.add(d_y_approx_2(5,h,x0),-d_y_exakt(5,x0))))
y_values_2=np.log(np.abs(np.add(d_y_approx_3(5,h,x0),-d_y_exakt(5,x0))))
y_expected1=x_values-x_values[0]+y_values_1[0]
y_expected2=x_values*2-x_values[0]*2+y_values_2[0]
plt.plot(x_values,y_expected1,label="erwartet 2punkt")
plt.plot(x_values,y_expected2,label="erwartet 3punkt")
plt.plot(x_values,y_values_1,label="2punkt")
plt.plot(x_values,y_values_2,label="3punkt")
plt.legend()
plt.xlabel(r"$\ln{h}$")
plt.ylabel(r"$\ln{\Delta_g}$")
plt.savefig("Verfahrensfehler_x2");
plt.show()
print("hmax für x2,2punkt und omega=5: "+str(get_h_max(x_values,y_values_1,y_expected1,1)))
print("hmax für x2,3punkt und omega=5: "+str(get_h_max(x_values,y_values_2,y_expected2,1)))
plt.xlabel(r"$\omega$")
plt.ylabel(r"$h_{max}$")
plt.plot(np.linspace(1,21,10000),get_h_max_array_2point(h,x_values,1,x0),label="2punkt")
plt.plot(np.linspace(1,21,10000),get_h_max_array_3point(h,x_values,1,x0),label="3punkt")
plt.legend()
plt.savefig("Hmax_von_omega_2")
plt.show()#




