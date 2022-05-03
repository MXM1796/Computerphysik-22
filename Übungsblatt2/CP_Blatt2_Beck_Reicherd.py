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
m= (average_xy-average_x*average_y)/(average_x2-average_x**2)# Berchnung der Geraden
n= (average_x2*average_y-average_x*average_xy)/(average_x2-average_x**2)
chi= np.average(np.power(np.add(y_vLog,np.add(np.multiply(-m,x_vLog),-n)),2))# Berechnung des mittleren Abstandsquadrats
print("m="+str(m)+",  n="+str(n),"durchschnittliches Abstandsquadrat:"+str(chi))
x_fit=np.arange(np.min(x_vLog),np.max(x_vLog),0.01)# Berechnen der Geradenpunkte
y_fit=x_fit*m+n
plt.xlabel(r"$\ln(h)$")# Beschriften
plt.ylabel(r"$\ln(\Delta_g)$")
plt.plot(x_fit,y_fit,label="Geradenfit")#plotten
plt.scatter(x_vLog,y_vLog,4,color='red',label="Datenpunkte")
plt.legend()
plt.savefig("Geradenfit_Aufgabenteil1")# Graphik speichern
plt.show()# Anzeigen der Graphik

##################### zweiter Geradenfit ##################
x_mLog=np.log(x_m) #Logarithmieren der Skalen
y_mLog=np.log(y_m)
average_x= np.average(x_mLog) # Berechnung verschiedener Mittelwerte
average_y= np.average(y_mLog)
average_x2= np.average(np.power(x_mLog,2))
average_xy= np.average(np.multiply(x_mLog,y_mLog))
m= (average_xy-average_x*average_y)/(average_x2-average_x**2)# Berchnung der Geraden
n= (average_x2*average_y-average_x*average_xy)/(average_x2-average_x**2)
chi= np.average(np.power(np.add(y_mLog,np.add(np.multiply(-m,x_mLog),-n)),2))# Berechnung des mittleren Abstandsquadrats
print("m="+str(m)+",  n="+str(n),"durchschnittliches Abstandsquadrat:"+str(chi))
x_fit=np.arange(np.min(x_mLog),np.max(x_mLog),0.01)# Berechnen der Geradenpunkte
y_fit=x_fit*m+n
plt.xlabel(r"$\ln(h)$")# Beschriften
plt.ylabel(r"$\ln(\Delta_g)$")
plt.plot(x_fit,y_fit,label="Geradenfit")#plotten
plt.scatter(x_mLog,y_mLog,4,color='red',label="Datenpunkte")
plt.legend()
plt.savefig("Geradenfit_Aufgabenteil1")# Graphik speichern
plt.show()# Anzeigen der Graphik


##################### Aufgabenteil zwei ###################
ZweiPi = 2 * np.pi
w_arr = np.arange(1, 21)
h = np.linspace(1, 2, 5)
print(h)


# Zwei Punkt Formel
def d_y_approx_2(w, h):
    d_y_a_3 = np.divide(np.cos(ZweiPi * (w + h) * (0.1 / (w + h))) - np.cos(ZweiPi * w * (0.1 / w)), ((w + h) - w))
    return d_y_a


def d_y_exakt(w):
    d_y_e = - (ZweiPi * w) * np.sin(ZweiPi * w * (0.1 / w))
    return d_y_e

# drei Punkt Formel
def d_y_approx_3(w, h):
    d_y_a_3 = np.divide(np.cos(ZweiPi * (w + h) * (0.1 / (w + h))) - np.cos(ZweiPi * (w - h) * (0.1 / (w - h))), 2*((w - h)- w))
    return d_y_a_3


def d_y_exakt(w):
    d_y_e = - (ZweiPi * w) * np.sin(ZweiPi * w * (0.1 / w))
    return d_y_e
