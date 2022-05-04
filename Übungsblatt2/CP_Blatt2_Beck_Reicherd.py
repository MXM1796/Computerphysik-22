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
print("Die Steigung und das durchschnittliche Abstandsquadrat für die Verfahrensfehlergerade")
print("m1="+str(m1)+",  n1="+str(n1),"durchschnittliches Abstandsquadrat:"+str(chi))
x_fit=np.arange(np.max(-12.0),np.max(x_vLog),0.01)# Berechnen der Geradenpunkte
y_fit=x_fit*m1+n1
plt.xlabel(r"$\ln(h)$")# Beschriften
plt.ylabel(r"$\ln(\Delta_g)$")
plt.plot(x_fit,y_fit,label="Geradenfit")#plotten
plt.scatter(x_vLog,y_vLog,4,color='red',label="Datenpunkte")
plt.legend()
plt.savefig("Geradenfit1")# Graphik speichern


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
print("Die Steigung und das durchschnittliche Abstandsquadrat für die Machinefehlergerade")
print("m2="+str(m2)+",  n2="+str(n2),"durchschnittliches Abstandsquadrat:"+str(chi))
x_fit=np.arange(np.min(x_mLog),-5.0,0.01)# Berechnen der Geradenpunkte
y_fit=x_fit*m2+n2
plt.xlabel(r"$\ln(h)$")# Beschriften
plt.ylabel(r"$\ln(\Delta_g)$")
plt.plot(x_fit,y_fit,label="Geradenfit")#plotten
plt.scatter(x_mLog,y_mLog,4,color='red')
plt.legend()
plt.savefig("Geradenfit2")# Graphik speichern
plt.show()# Anzeigen der Graphik

hop = np.exp(np.divide(n1-n2,m2-m1))

print("Der Schnittpunkt liegt bei",round(hop,5))


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
