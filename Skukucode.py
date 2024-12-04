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


##########################################
#Duration change between fires
w = .5
v = .1


##########################################



# Function to capitalize lines that start with '&'
def capitalize_ampersand_strings(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    with open(output_file, 'w') as file:
        for line in lines:
            words = line.split()
            updated_words = [word.upper() if word.startswith('&') else word for word in words]
            file.write(" ".join(updated_words) + "\n")

def sum_HRR_per_pixel(burner):
    act_pixels = np.where((burner.fre_f > 0) | (burner.fre_s > 0))
    
    hrr_act_pix = []
    
    total_HRR = 0
    
    for i, j in zip(*act_pixels):
        hrr_value = float((burner.fre_f[i, j] + burner.fre_s[i, j])*1.e3)
        burner_id = f"Burner_{i}_{j}"
        hrr_act_pix.append((burner_id, hrr_value))
        # Acumular la suma total de HRR
        total_HRR += hrr_value

    return hrr_act_pix, total_HRR


###########################################
if __name__ == '__main__':
###########################################

    mm = np.load('skukuza4_4ForeFire.npy')
    subset_size = 15
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
    x = np.arange(0, subset_size, 1)
    y = np.arange(0, subset_size, 1)
    grid_n, grid_e = np.meshgrid(y,x)

    burner = np.zeros((subset_size, subset_size), dtype=np.dtype([('fre_f',float),('fre_s',float),('residenceTime',float),('burningTime',float),('arrivalTime',float),('grid_e',float),('grid_n',float)]) ) 
    burner = subset.view(np.recarray)
    burner.grid_e = grid_e
    burner.grid_n = grid_n


    # Create the Burner object with the original templates
    template = Burner(nml)

    act_pixels = np.where((burner.fre_f > 0) | (burner.fre_s > 0))

    xmin = grid_e
    xmax = xmin + 1
    ymin = grid_n
    ymax = ymin + 1

    
    for i,j in zip(*act_pixels):

	    print(i,j)
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
	    
	    
	    template_here = template.copy()

	    # Actualización de los parámetros en el template
	    template_here.surf_template['hrrpua'] = HRRPUA_f  
	    template_here.surf_template['id'] = surf_id
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
	    #template_here.vent_template['xb'] = np.round(np.array([xmin[i], xmax[i], ymin[i], ymax[i], 0.0, 0.0]), 3)
	    template_here.vent_template['xb'] = np.round(np.array([grid_e[i, j], grid_e[i, j] + 1, grid_n[i, j], grid_n[i, j] + 1,0.0, 0.0]), 3)
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

    hrr_act_pix, total_HRR=sum_HRR_per_pixel(burner)
    print(f"La suma total de HRR es: {total_HRR:.4f} KJ")