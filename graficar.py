import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage
import f90nml
import sys, os
import copy
import pandas as pd
from scipy import integrate
from numpy import trapz

###########################################
class Burner:
    def __init__(self,nml):
        self.surf_template     =  nml['SURF']
        self.ramp_template     =  nml['ramp']
        self.vent_template     =  nml['vent'][0]

    def copy(self):
        return  copy.deepcopy(self)
###########################################
###############################################
Rf_f = .10 #Radioactive fraction of flaming
Rf_s = .3  #Radioactive fraction of smoldering
dt = 1     
v = .1 #Duration change between times
w = .5 #Duration change between times
offset= 60	#temps de començament de l'incendi
###############################################

# Function to capitalize lines that start with '&'
def capitalize_ampersand_strings(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    with open(output_file, 'w') as file:
        for line in lines:
            words = line.split()
            updated_words = [word.upper() if word.startswith('&') else word for word in words]
            file.write(" ".join(updated_words) + "\n")

graficados = {}
'''
def frp(burner, i, j, t):
    #if (burner.fre_f[i, j] <= 0) and (burner.fre_s[i, j] <= 0):
        #print(f"El píxel ({i},{j}) no tiene valores de HRR.")
        #return    

    arrivalT = burner.arrivalTime[i, j]
    residenceT = burner.residenceTime[i, j]
    burningT = burner.burningTime[i, j]
    dx = 1  # Se asume tamaño de celda 1x1
    dy = 1

    # Cálculo de HRRPUA
    HRRPUA_f = round(burner.fre_f[i, j] / (Rf_f * (w / 2 + residenceT) * dx * dy * 1.e3), 4) if burner.fre_f[i,j] > 0 else 0
    HRRPUA_s = round(burner.fre_s[i, j] / (Rf_s * (burningT - residenceT) * dx * dy * 1.e3), 4) if burner.fre_s[i,j] > 0 else 0

    # Definir la curva de HRR en función del tiempo
    tiempos = []
    HRR_values = []

    for tiempo_iter in range(int(arrivalT.max() + burningT.max() + 10)):  # Hasta el final del quemado
        if tiempo_iter <= arrivalT:
            HRR = 0
        elif arrivalT < tiempo_iter <= arrivalT + residenceT:
            HRR = HRRPUA_f
        elif arrivalT + residenceT < tiempo_iter <= arrivalT + burningT:
            HRR = HRRPUA_s
        else:
            HRR = 0

        tiempos.append(tiempo_iter)
        HRR_values.append(HRR)

    # Solo graficar si el píxel no ha sido graficado antes
    if (i, j) not in graficados:
        plt.figure(figsize=(8, 5))
        plt.title(f'HRR en función del tiempo - Píxel ({i},{j})')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('HRR (kW)')
        plt.plot(tiempos, HRR_values, color='red', label=f'HRR ({i},{j})')
        plt.legend()
        plt.grid(True)
        plt.show()
        graficados[(i, j)] = True  # Marcar como graficado

    # Buscar el valor de HRR en el tiempo solicitado
    if t in tiempos:
        indice = tiempos.index(t)
        print(f"HRR en ({i}, {j}) en t={t}s: {HRR_values[indice]} kW")
    else:
        print(f"No hay valor de HRR en t={t}s para el píxel ({i}, {j}).")
    
    return tiempos, HRR_values
'''
import numpy as np
'''
def frp(burner, i, j, t):
    arrivalT = burner.arrivalTime[i, j]
    residenceT = burner.residenceTime[i, j]
    burningT = burner.burningTime[i, j]
    dx = 1  # Se asume tamaño de celda 1x1
    dy = 1

    # Cálculo de HRRPUA
    HRRPUA_f = round(burner.fre_f[i, j] / (Rf_f * (w / 2 + residenceT) * dx * dy * 1.e3), 4) if burner.fre_f[i, j] > 0 else 0
    HRRPUA_s = round(burner.fre_s[i, j] / (Rf_s * (burningT - residenceT) * dx * dy * 1.e3), 4) if burner.fre_s[i, j] > 0 else 0

    # Definir la curva de HRR en función del tiempo
    tiempos = []
    HRR_values = []
    HRR_tiempo = []  # Lista para almacenar HRR de todos los píxeles en cada segundo

    tiempo_max = int(arrivalT.max() + burningT.max() + 10)

    for tiempo_iter in range(tiempo_max):  # Hasta el final del quemado
        if tiempo_iter <= arrivalT:
            HRR = 0
        elif arrivalT < tiempo_iter <= arrivalT + residenceT:
            HRR = HRRPUA_f
        elif arrivalT + residenceT < tiempo_iter <= arrivalT + burningT:
            HRR = HRRPUA_s
        else:
            HRR = 0

        tiempos.append(tiempo_iter)
        HRR_values.append(HRR)

    # Solo graficar si el píxel no ha sido graficado antes
    if (i, j) not in graficados:
        plt.figure(figsize=(8, 5))
        plt.title(f'HRR en función del tiempo - Píxel ({i},{j})')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('HRR (kW)')
        plt.plot(tiempos, HRR_values, color='red', label=f'HRR ({i},{j})')
        plt.legend()
        plt.grid(True)
        plt.show()
        graficados[(i, j)] = True  # Marcar como graficado

    # Buscar el valor de HRR en el tiempo solicitado
    if t in tiempos:
        indice = tiempos.index(t)
        print(f"HRR en ({i}, {j}) en t={t}s: {HRR_values[indice]} kW")
    else:
        print(f"No hay valor de HRR en t={t}s para el píxel ({i}, {j}).")
    
    return tiempos, HRR_values
'''
def frp(burner, i, j, t):
    arrivalT = burner.arrivalTime[i, j]
    residenceT = burner.residenceTime[i, j]
    burningT = burner.burningTime[i, j]
    dx = 1  # Se asume tamaño de celda 1x1
    dy = 1

    # Cálculo de HRRPUA
    HRRPUA_f = round(burner.fre_f[i, j] / (Rf_f * (w / 2 + residenceT) * dx * dy * 1.e3), 4) if burner.fre_f[i, j] > 0 else 0
    HRRPUA_s = round(burner.fre_s[i, j] / (Rf_s * (burningT - residenceT) * dx * dy * 1.e3), 4) if burner.fre_s[i, j] > 0 else 0

    # Definir la curva de HRR en función del tiempo
    tiempos = []
    HRR_values = []
    HRR_tiempo = []  # Lista para almacenar HRR de todos los píxeles en cada segundo

    tiempo_max = int(arrivalT.max() + burningT.max() + 10)

    for tiempo_iter in range(tiempo_max):  # Hasta el final del quemado
	    if 'arrivalT' in locals() and 'residenceT' in locals() and 'burningT' in locals():
	        if tiempo_iter <= arrivalT:
	            HRR = 0
	        elif arrivalT < tiempo_iter <= arrivalT + residenceT:
	            HRR = HRRPUA_f
	        elif arrivalT + residenceT < tiempo_iter <= arrivalT + burningT:
	            HRR = HRRPUA_s
	        else:
	            HRR = 0
	    else:
	        HRR = 0  # Si no hay tiempos definidos, HRR es 0 en toda la simulación

	    tiempos.append(tiempo_iter)
	    HRR_values.append(HRR)

	# Solo graficar si el píxel no ha sido graficado antes
    if (i, j) not in graficados:
	    plt.figure(figsize=(8, 5))
	    plt.title(f'HRR en función del tiempo - Píxel ({i},{j})')
	    plt.xlabel('Tiempo (s)')
	    plt.ylabel('HRR (kW)')
	    plt.plot(tiempos, HRR_values, color='red', label=f'HRR ({i},{j})')
	    plt.legend()
	    plt.grid(True)
	    plt.show()
	    graficados[(i, j)] = True  # Marcar como graficado

	# Buscar el valor de HRR en el tiempo solicitado
    if t in tiempos:
	    indice = tiempos.index(t)
	    print(f"HRR en ({i}, {j}) en t={t}s: {HRR_values[indice]} kW")
    else:
	    print(f"No hay valor de HRR en t={t}s para el píxel ({i}, {j}).")

    return tiempos, HRR_values
###########################################
if __name__ == '__main__':
###########################################

    # Input directory and configuration file
    inputDir = './'
    fileConfigIn = inputDir+'input_fixed_burner.fds'
    output_csv = '~/Src/FdsBurner/fixed_burner_devc.csv'

    # Read the .fds file and retrieve SURF, RAMP, and VENT templates
    nml = f90nml.read(fileConfigIn)
    surf_templates_original = nml['surf']
    ramp_templates_original = nml['ramp']
    vent_templates_original = nml['vent']

    # Remove unnecessary blocks
    del nml['TAIL'], surf_templates_original, ramp_templates_original, vent_templates_original
    
    # Remove old input file if it exists
    if os.path.isfile(inputDir+'input_mynewInput.fds'):
        os.remove(inputDir+'input_mynewInput.fds')

    # Define the burner matrix
    # Grid of the input data
    x = np.arange(0, 4, 1)
    y = np.arange(0, 4, 1)
    grid_n, grid_e = np.meshgrid(y,x)

    burner = np.zeros((4, 4), dtype=np.dtype([('fre_f',float),('fre_s',float),('residenceTime',float),('burningTime',float),('arrivalTime',float),('grid_e',float),('grid_n',float)]) ) 
    burner = burner.view(np.recarray)
    burner.grid_e = grid_e
    burner.grid_n = grid_n
    
    burner.fre_f[:,:] = -999
    burner.fre_s[:,:] = -999  

    burner.fre_f[0:1,0:1] = 0
    burner.fre_s[0:1,0:1] = -100
    
    burner.arrivalTime[0:1,0:1]= 5
    burner.residenceTime[0:1,0:1]= 10
    burner.burningTime[0:1,0:1]= 25  

    burner.fre_f[1:2, 1:2] = 75500
    burner.fre_s[1:2, 1:2]= 12750
    
    burner.arrivalTime[1:2, 1:2]= 5
    burner.residenceTime[1:2, 1:2]= 10
    burner.burningTime[1:2, 1:2]= 25  
    ##################################
    burner.fre_f[1:2, 2:3] = 50000
    burner.fre_s[1:2, 2:3] = 12750
    
    burner.arrivalTime[1:2, 2:3]= 5
    burner.residenceTime[1:2, 2:3]= 10
    burner.burningTime[1:2, 2:3]= 35    
    ##################################
    burner.fre_f[2:3, 1:2] = 30500
    burner.fre_s[2:3, 1:2] = 12750
    
    burner.arrivalTime[2:3, 1:2]= 5
    burner.residenceTime[2:3, 1:2]= 10
    burner.burningTime[2:3, 1:2]= 25   
    ##################################
    burner.fre_f[2:3,2:3] = 25500
    burner.fre_s[2:3,2:3]  = 12750
    
    burner.arrivalTime[2:3,2:3] = 5
    burner.residenceTime[2:3,2:3] = 10
    burner.burningTime[2:3,2:3] = 25      
    


    colors = ["ORANGE", "RED", "BLUE", "GREEN", "YELLOW"]

    # Create the Burner object with the original templates
    template = Burner(nml)


    act_pixels = np.where((burner.fre_f > 0) | (burner.fre_s > 0))

    xmin = grid_e[act_pixels]
    xmax = xmin + 1
    ymin = grid_n[act_pixels]
    ymax = ymin + 1
    
    t_s = 0 + offset
    t_e = int(burner.arrivalTime.max() + burner.burningTime.max() + offset)

    for idx, (i,j) in enumerate(zip(*act_pixels)):
        arrivalT = burner.arrivalTime[i,j]
        residenceT = burner.residenceTime[i,j]
        burningT = burner.burningTime[i,j]
        dx=x[i+1]-x[i]
        dy=y[j+1]-y[j]
        HRRPUA_f= round(burner.fre_f[i,j]/(Rf_f*(w/2+residenceT)*dx*dy*1.e3),4)
        HRRPUA_s= round(burner.fre_s[i,j]/(Rf_s*(burningT-residenceT)*dx*dy*1.e3),4)
        surf_id = f'BURNER_{i}_{j}'
        ramp_id = f'ramp__{i}_{j}'
        color = colors[idx % len(colors)]

        template_here = template.copy()

        template_here.surf_template['hrrpua'] = HRRPUA_f  
        template_here.surf_template['id'] = surf_id
        template_here.surf_template['color'] = color
        template_here.surf_template['ramp_q'] = ramp_id

        for k in range(7):  
            template_here.ramp_template[k]['id'] = ramp_id
        template_here.ramp_template[1]['t'] = arrivalT - w

        template_here.ramp_template[2]['t'] = arrivalT
        template_here.ramp_template[2]['f'] = HRRPUA_f/HRRPUA_f
        
        template_here.ramp_template[3]['t'] = arrivalT + residenceT
        template_here.ramp_template[3]['f'] = HRRPUA_f/HRRPUA_f

        template_here.ramp_template[4]['t'] = arrivalT + residenceT + v
        template_here.ramp_template[4]['f'] = HRRPUA_s/HRRPUA_f

        template_here.ramp_template[5]['t'] = arrivalT + burningT
        template_here.ramp_template[5]['f'] = HRRPUA_s/HRRPUA_f

        template_here.ramp_template[6]['t'] = arrivalT + burningT + v


        template_here.vent_template['xb'] = np.round([xmin[idx], xmax[idx], ymin[idx], ymax[idx], 0.0, 0.0], 3)
        template_here.vent_template['surf_id'] = surf_id

        # Add the new blocks to the NML file
        nml.add_cogroup('SURF', template_here.surf_template)
        [nml.add_cogroup('RAMP', ramp_) for ramp_ in template_here.ramp_template]
        nml.add_cogroup('VENT', template_here.vent_template)

    # Add the &TAIL at the end
    nml['tail'] = {}

    # Remove the initial SURF, RAMP, and VENT blocks
    del nml['SURF'][0]
    for i in range(7):
        del nml['RAMP'][0]
    del nml['VENT'][0]

    f90nml.write(nml, 'tmp.fds')

    capitalize_ampersand_strings('tmp.fds', inputDir+os.path.basename(fileConfigIn).split('.')[0]+'_withBurner.fds')    
    os.remove('tmp.fds')
frp(burner,0,0,15)
frp(burner,2,1,15)
frp(burner,2,1,16)
frp(burner,2,1,15.5)
frp(burner,2,2,15)