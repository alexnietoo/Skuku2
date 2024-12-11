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
import multiprocessing as mp

###########################################
class Burner:
    def __init__(self, nml):
        self.surf_template = nml['SURF']
        self.ramp_template = nml['ramp']
        self.vent_template = nml['vent'][0]

    def copy(self):
        return copy.deepcopy(self)
###########################################

# Duration change between fires
w = .5
v = .1
Rf_f = .10 #Radioactive fraction of flaming
Rf_s = .3  #Radioactive fraction of smoldering
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

def sum_HRR_per_pixel(burner):
    act_pixels = np.where((burner.fre_f > 0) | (burner.fre_s > 0))
    
    hrr_act_pix = []
    total_HRR = 0
    
    for i, j in zip(*act_pixels):
        hrr_value = float((burner.fre_f[i, j] + burner.fre_s[i, j])*1.e3)
        burner_id = f"Burner_{i}_{j}"
        hrr_act_pix.append((burner_id, hrr_value))
        total_HRR += hrr_value

    return hrr_act_pix, total_HRR

def process_pixel(args):
    for i,j in zip(*act_pixels):
        
        i, j, subset, template, x, y, w, v = args

        arrivalT = subset[i, j]['arrivalTime']-515
        residenceT = subset[i, j]['residenceTime']
        burningT = subset[i, j]['burningTime']
        fre_f = subset[i, j]['fre_f']
        fre_s = subset[i, j]['fre_s']
        #if i < len(x) - 1: 
            #dx = x[i+1] - x[i]
        #else: 1
        dx = 1
        dy = 1
        #if j < len(y) - 1: 
            #dy = y[j+1] - y[j]
        #else: 1


        if residenceT * dx * dy == 0:
            HRRPUA_f = 0
        else:
            HRRPUA_f = round(fre_f/Rf_f * 1.e3 / (residenceT * dx * dy), 10)

        if (burningT - residenceT) * dx * dy <= 0:  
            HRRPUA_s = 0
        else:
            HRRPUA_s = round(fre_s/Rf_s * 1.e3 / ((burningT - residenceT) * dx * dy), 10)


        surf_id = f'BURNER_{i}_{j}'
        ramp_id = f'ramp__{i}_{j}'
        
        template_here = template.copy()

        template_here.surf_template['hrrpua'] = HRRPUA_f  
        template_here.surf_template['id'] = surf_id
        template_here.surf_template['ramp_q'] = ramp_id

        ramp_f_f = HRRPUA_f / HRRPUA_f if HRRPUA_f != 0 else 0
        ramp_s_f = HRRPUA_s / HRRPUA_f if HRRPUA_f != 0 else 0
        
        for k in range(7):
            template_here.ramp_template[k]['id'] = ramp_id
        template_here.ramp_template[1]['t'] = arrivalT - w
        template_here.ramp_template[2]['t'] = arrivalT
        template_here.ramp_template[2]['f'] = ramp_f_f
        template_here.ramp_template[3]['t'] = arrivalT + residenceT
        template_here.ramp_template[3]['f'] = ramp_f_f
        template_here.ramp_template[4]['t'] = arrivalT + residenceT + v
        template_here.ramp_template[4]['f'] = ramp_s_f
        template_here.ramp_template[5]['t'] = arrivalT + burningT
        template_here.ramp_template[5]['f'] = ramp_s_f
        template_here.ramp_template[6]['t'] = arrivalT + burningT + v

        template_here.vent_template['xb'] = np.round(np.array([i, i + 1, j, j + 1, 0.0, 0.0]), 3)
        template_here.vent_template['surf_id'] = surf_id
        print (i,j)
        return template_here


###########################################
if __name__ == '__main__':
###########################################

    mm = np.load('skukuza4_4ForeFire.npy')
    subset_size = 96
    start = (mm.shape[0]-subset_size)//2
    end = start + subset_size
    subset = mm[start:end, start:end]
    np.save("mid_subset.npy", subset)

    inputDir = './'
    fileConfigIn = inputDir'input_fixed_burner.fds'
    output_csv = '~/Src/FdsSkuku/fixed_burner_devc.csv'

    nml = f90nml.read(fileConfigIn)
    surf_templates_original = nml['surf']
    ramp_templates_original = nml['ramp']
    vent_templates_original = nml['vent']
    del nml['TAIL'], surf_templates_original, ramp_templates_original, vent_templates_original
    
    if os.path.isfile(inputDir+'input_mynewInput.fds'):
        os.remove(inputDir+'input_mynewInput.fds')

    x = np.arange(0, subset_size, 1)
    y = np.arange(0, subset_size, 1)
    grid_n, grid_e = np.meshgrid(y, x)

    burner = subset.view(np.recarray)
    burner.grid_e = grid_e
    burner.grid_n = grid_n

    template = Burner(nml)
    act_pixels = np.where((burner.fre_f > 0) | (burner.fre_s > 0))
    args = [(i, j, subset, template, x, y, w, v) for i, j in zip(*act_pixels)]


    # Verificar si $SLURM_NTASKS existe
    if 'SLURM_NTASKS' in os.environ:
        # Leer el valor y convertirlo a un entero
        ntasks = int(os.getenv('SLURM_NTASKS'))
        print(f"SLURM_NTASKS existe y su valor es: {slurm_ntasks}")
    else:
        # Acción alternativa si no existe
        ntasks = mp.cpu_count()
        print("SLURM_NTASKS no está definido en las variables de entorno.")


    # Use multiprocessing to process pixels
    with mp.Pool(ntasks) as pool:
        results = pool.map(process_pixel, args)

    # Add processed templates to the .fds file
    for template_here in results:
        nml.add_cogroup('SURF', template_here.surf_template)
        [nml.add_cogroup('RAMP', ramp_) for ramp_ in template_here.ramp_template]
        nml.add_cogroup('VENT', template_here.vent_template)

    nml['tail'] = {}
    del nml['SURF'][0]
    for i in range(7):
        del nml['RAMP'][0]
    del nml['VENT'][0]

    f90nml.write(nml, 'tmp.fds')
    capitalize_ampersand_strings('tmp.fds', inputDir+os.path.basename(fileConfigIn).split('.')[0]+'_withBurner.fds')    
    os.remove('tmp.fds')

    hrr_act_pix, total_HRR = sum_HRR_per_pixel(burner)
    print(f"La suma total de HRR es: {total_HRR:.4f} KJ")
