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

# Function to capitalize lines that start with '&'
def capitalize_ampersand_strings(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    with open(output_file, 'w') as file:
        for line in lines:
            words = line.split()
            updated_words = [word.upper() if word.startswith('&') else word for word in words]
            file.write(" ".join(updated_words) + "\n")

def plot_column(column_name):
    df = pd.read_csv('./fixed_burner_devc.csv', header = 1)
    df.columns = ['Time', 'HRR', 'Temp']
    df = df.set_index('Time')
    df[column_name].plot()
    plt.title(f'{column_name} over Time')
    plt.xlabel('Time')
    plt.ylabel(column_name)
    plt.show()
#################
def plot_from_multiple_files(hrr_csv, devc_csv):
    # Leer los datos del archivo CSV para HRR
    df_hrr = pd.read_csv(hrr_csv, header=1)
    df_hrr.columns = ['Time', 'HRR','Q_RADI','Q_CONV','Q_COND','Q_DIFF','Q_PRES','Q_PART','Q_ENTH','Q_TOTAL','MLR_AIR','MLR_CELLULOSE','MLR_PRODUCTS']
    df_hrr = df_hrr.set_index('Time')  
    
    
    # Leer los datos del archivo CSV para devc
    df_devc = pd.read_csv(devc_csv, header=1)
    df_devc.columns = ['Time', 'HRR', 'Temp']  
    df_devc = df_devc.set_index('Time')  
    volume = 25e-6
    df_devc['HRR'] = df_devc['HRR'] * volume


    # Graficar HRR del archivo hrr_csv y del devc_csv
    df_hrr['HRR'].plot(label='HRR from HRR CSV', color='blue')
    
    df_devc['HRR'].plot(label='HRR from devc CSV', color='black')

    # Configuración de la gráfica
    plt.title(f'HRR Comparison whole domain and devc')
    plt.xlabel('Time (s)')
    plt.ylabel('HRR (kW)')
    plt.legend()  # Mostrar la leyenda
    plt.grid(True)  # Mostrar una cuadrícula para mejor lectura
    plt.show()

 
    ##################

'''
def E_devc(column_name):
    df = pd.read_csv('./fixed_burner_devc.csv', header = 1)
    time = df.index.values 
    values = df[column_name].values
    area_total = 0  
    
    for i in range(len(time) - 1):
        delta_t = time[i+1] - time[i] 
        base = values[i]  
        area = delta_t * base  
        area_total += area  
    area_total_r = area_total*(25.e-6)
    
    print(f'The Energy recieved at the devc from {column_name} is: {area_total_r:.4f} KJ')
'''
'''
def E_domain(column_name):
    df = pd.read_csv('./fixed_burner_hrr.csv', header = 1)
    time = df.index.values 
    values = df['HRR'].values
    area_total = np.trapezoid(values, time)

    print(f'The Domain Energy from {column_name} is: {area_total:.4f} KJ')
'''
def E_domain(column_name):
    df1 = pd.read_csv('./fixed_burner_hrr.csv', header = 1)
    df1.columns = ['Time', 'HRR', 'Q_RADI', 'Q_CONV', 'Q_COND', 'Q_DIFF', 'Q_PRES', 'Q_PART', 'Q_ENTH', 'Q_TOTAL', 'MLR_AIR', 'MLR_CELLULOSE', 'MLR_PRODUCTS']
    tim = df1['Time'].values 
    values = df1[column_name].values
    area_total = 0  
      
    for i in range(len(tim) - 1):
        delta_t = tim[i+1] - tim[i] 
        base = values[i]  
        area = delta_t * base  
        area_total += area  
    
    print(f'The Domain Energy from {column_name} is: {area_total:.4f} KJ')
def sum_HRR_per_pixel(burner):
    act_pixels = np.where((burner.fre_f > 0) | (burner.fre_s > 0))
    
    hrr_act_pix = []
    
    total_HRR = 0
    
    for i, j in zip(*act_pixels):
        hrr_value = float((burner.fre_f[i, j] + burner.fre_s[i, j])*1.e-3)
        burner_id1 = f"Burner_{i}_{j}"
        hrr_act_pix.append((burner_id1, hrr_value))
        # Acumular la suma total de HRR
        total_HRR += hrr_value

    return hrr_act_pix, total_HRR



###########################################
if __name__ == '__main__':
###########################################
    mm = np.load('skukuza4_4ForeFire.npy')
    subset_size = 96
    start = (mm.shape[0]-subset_size)//2
    end = start + subset_size
    subset= mm[start:end,start:end]
    np.save("mid_subset.npy",subset)

    # Input directory and configuration file
    inputDir = '/home/alex/Data/FDS/FixedBurner/'
    fileConfigIn = inputDir+'input_fixed_burner.fds'
    output_csv = '~/Src/FdsSkuku/fixed_burner_devc.csv'

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
    x = np.arange(0, 96, 1)
    y = np.arange(0, 96, 1)
    grid_n, grid_e = np.meshgrid(y,x)

    burner = np.zeros((96, 96), dtype=np.dtype([('fre_f',float),('fre_s',float),('residenceTime',float),('burningTime',float),('arrivalTime',float),('grid_e',float),('grid_n',float)]) ) 
    burner = subset.view(np.recarray)
    burner.grid_e = grid_e
    burner.grid_n = grid_n
    '''
    burner.fre_f[:,:] = -9999
    burner.fre_f[15:16, 15:16] = 25500
    burner.fre_s[:,:] = -9999
    burner.fre_s[15:16, 15:16] = 12750

    burner.fre_f[4:5, 4:5] = 25500
    burner.fre_s[4:5, 4:5] = 12750
    
    burner.arrivalTime[15:16, 15:16] = 5
    burner.residenceTime[15:16, 15:16] = 10
    burner.burningTime[15:16, 15:16] = 25

    burner.arrivalTime[4:5, 4:5] = 5
    burner.residenceTime[4:5, 4:5] = 10
    burner.burningTime[4:5, 4:5] = 25
    '''

    
    #colors = ["ORANGE", "RED", "BLUE", "GREEN", "YELLOW"]

    # Create the Burner object with the original templates
    template = Burner(nml)

    act_pixels = np.where((burner.fre_f > 0) | (burner.fre_s > 0))

    xmin = grid_e[act_pixels]
    xmax = xmin + 1
    ymin = grid_n[act_pixels]
    ymax = ymin + 1

    #Duration change between fires
    v = .1
    w = .5
    for idx, (i,j) in enumerate(zip(*act_pixels)):
    # Recorre todos los elementos del array con un solo loop
    for idx in range(subset.size):
        # Convierte el índice lineal a índices de fila y columna
        i, j = np.unravel_index(idx, subset.shape)
    
        # Verifica si el píxel es activo utilizando 'plotMask' o algún otro criterio
        if subset[i, j]['plotMask'] > 0:  # Asegúrate de que 'plotMask' indique si el píxel está activo
         # Extrae los valores directamente del array
            arrivalT = subset[i, j]['arrivalTime']
            residenceT = subset[i, j]['residenceTime']
            burningT = subset[i, j]['burningTime']
            fre_f = subset[i, j]['fre_f']
            fre_s = subset[i, j]['fre_s']

            if i < len(x) - 1: 
                dx = x[i+1] - x[i]

            if j < len(y) - 1: 
                dy = y[j+1] - y[j]
            
            # Calcula HRRPUA utilizando los valores extraídos del array
            HRRPUA_f = round(fre_f*1.e3 / ((w + residenceT) * dx * dy), 10)
            HRRPUA_s = round(fre_s*1.e3 / ((burningT - residenceT) * dx * dy), 10)
            
            surf_id = f'BURNER_{i}_{j}'
            ramp_id = f'ramp__{i}_{j}'
            #color = colors[(i * array.shape[1] + j) % len(colors)]
            
            template_here = template.copy()

            # Actualización de los parámetros en el template
            template_here.surf_template['hrrpua'] = HRRPUA_f  
            template_here.surf_template['id'] = surf_id
            #template_here.surf_template['color'] = color
            template_here.surf_template['ramp_q'] = ramp_id
            
            # Configuración del RAMP
            for k in range(7):
                template_here.ramp_template[k]['id'] = ramp_id
            template_here.ramp_template[1]['t'] = arrivalT - w
            template_here.ramp_template[2]['t'] = arrivalT
            template_here.ramp_template[2]['f'] = HRRPUA_f / HRRPUA_f
            template_here.ramp_template[3]['t'] = arrivalT + residenceT
            template_here.ramp_template[3]['f'] = HRRPUA_f / HRRPUA_f
            template_here.ramp_template[4]['t'] = arrivalT + residenceT + v
            template_here.ramp_template[4]['f'] = HRRPUA_s / HRRPUA_f
            template_here.ramp_template[5]['t'] = arrivalT + burningT
            template_here.ramp_template[5]['f'] = HRRPUA_s / HRRPUA_f
            template_here.ramp_template[6]['t'] = arrivalT + burningT + v

            # Configuración del VENT
            template_here.vent_template['xb'] = np.round(np.array([xmin[i], xmax[i], ymin[i], ymax[i], 0.0, 0.0]), 3)
            template_here.vent_template['surf_id'] = surf_id

            # Añadir bloques al archivo NML
            nml.add_cogroup('SURF', template_here.surf_template)
            [nml.add_cogroup('RAMP', ramp_) for ramp_ in template_here.ramp_template]
            nml.add_cogroup('VENT', template_here.vent_template)

    # Agregar el &TAIL al final
    nml['tail'] = {}
    # Remove the initial SURF, RAMP, and VENT blocks
    del nml['SURF'][0]
    for i in range(7):
        del nml['RAMP'][0]
    del nml['VENT'][0]

    f90nml.write(nml, 'tmp.fds')

    capitalize_ampersand_strings('tmp.fds', inputDir+os.path.basename(fileConfigIn).split('.')[0]+'_withBurner.fds')    
    os.remove('tmp.fds')

    #E_devc('HRR')
    E_domain('HRR')
    ## Llamada a la función para cada columna
    #plot_column('HRR')
    plot_from_multiple_files(hrr_csv='./fixed_burner_hrr.csv', devc_csv='./fixed_burner_devc.csv')
    #plot_column('Temp')
    hrr_act_pix, total_HRR = sum_HRR_per_pixel(burner)
    for burner_id1, hrr_value in hrr_act_pix:
        print(f"Suma de HRR por píxel-> {burner_id1}: {hrr_value:.4f} KJ")
    print(f"Suma total de HRR en todos los píxeles: {total_HRR:.4f} KJ")

    #carpeta_destino = os.path.expanduser("~/Src/FdsBurner")
    #nombre_archivo = "fixed_burner_devc.csv"
    #ruta_completa = os.path.join(carpeta_destino, nombre_archivo)
    #df.to_csv(ruta_completa,index= False)


