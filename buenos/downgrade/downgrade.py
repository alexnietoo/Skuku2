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

#######################################################################################################
def downgrade_resolution_4nadir(arr, diag_res_cte_shape, flag_interpolation='conservative'):

    '''
    flag_interpolation is conservative, max, min, average or sum 
    '''
    factor = (1.*arr.shape[0]/diag_res_cte_shape[0])
    if factor == int(np.floor(factor)): factor = int(factor)
    else: factor = int(factor) + 1
    #factor = int(np.round(old_div(1.*arr.shape[0],diag_res_cte_shape[0]),0))
    
    if np.mod( arr.shape[0], factor )!=0:
        extra_pixel0 = factor-np.mod( arr.shape[0], factor )
        extra_pixel1 = factor-np.mod( arr.shape[1], factor )
    else: 
        extra_pixel0 = 0
        extra_pixel1 = 0
  
    if (extra_pixel0>0) |  (extra_pixel1>0):
        x = np.arange(0,arr.shape[0],1)
        y = np.arange(0,arr.shape[1],1)
        z = arr.flatten()
        f = interpolate.interp2d(x, y, z, kind='linear')
        
        grid_x = np.arange(0-int(0.5*extra_pixel0),extra_pixel0-int(0.5*extra_pixel0)+arr.shape[0],1)
        grid_y = np.arange(0-int(0.5*extra_pixel1),extra_pixel1-int(0.5*extra_pixel1)+arr.shape[1],1) 
        arr = f(grid_x, grid_y)
        arr = arr.T
    
    if flag_interpolation == 'max':
        return shrink_max(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
   
    elif flag_interpolation == 'min':
        return shrink_min(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
    
    elif flag_interpolation == 'conservative':
        
        mask = np.where(arr!=-999, 1, 0)
        sum_pixel = shrink_sum(mask, diag_res_cte_shape[0], diag_res_cte_shape[1]) 
        
        sum = shrink_sum(arr, diag_res_cte_shape[0], diag_res_cte_shape[1]) 
        return np.where(sum != -999, (sum/sum_pixel), sum)

    elif flag_interpolation == 'average':
        return shrink_average(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
    
    elif flag_interpolation == 'sum':
        return shrink_sum(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])

    else:
        print('bad flag')
        pdb.set_trace()


###################################################################################################
def shrink_sum(data, nx, ny, nodata=-999):
    
    data_masked = np.ma.array(data, mask = np.where(data==nodata,1,0))
    nnx, nny    = data_masked.shape

    # Reshape data
    rshp = data_masked.reshape([nx, nnx//nx, ny, nny//ny])

    # Compute mean along axis 3 and remember the number of values each mean
    # was computed from
    return np.where(rshp.sum(3).sum(1).mask==False, rshp.sum(3).sum(1).data, nodata)

#######################################################################################################
def shrink_max(data, nx, ny, nodata=-999):
    data_masked = np.ma.array(data, mask = np.where(data==nodata,1,0))
    nnx, nny    = data_masked.shape

    # Reshape data
    rshp = data_masked.reshape([nx, nnx//nx, ny, nny//ny])

    # Compute mean along axis 3 and remember the number of values each mean
    # was computed from
    return np.where(rshp.max(3).max(1).mask==False, rshp.max(3).max(1).data, nodata)
    
    #return min3    return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).max(axis=1).max(axis=2)


#######################################################################################################
def shrink_min(data, nx, ny, nodata=-999):
    
    data_masked = np.ma.array(data, mask = np.where(data==nodata,1,0))
    nnx, nny    = data_masked.shape

    # Reshape data
    rshp = data_masked.reshape([nx, nnx//nx, ny, nny//ny])

    # Compute mean along axis 3 and remember the number of values each mean
    # was computed from
    return np.where(rshp.min(3).min(1).mask == False, rshp.min(3).min(1).data, nodata)

    #return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).min(axis=1).min(axis=2)


#######################################################################################################
def shrink_average(data, nx, ny, nodata=-999.):
   
    data_masked = np.ma.array(data, mask = np.where(data==nodata, 1, 0))
    nnx, nny    = data_masked.shape

    # Reshape data
    rshp = data_masked.reshape([nx, nnx//nx, ny, nny//ny])

    # Compute mean along axis 3 and remember the number of values each mean
    # was computed from
    mean3 = rshp.mean(3)
    count3 = rshp.count(3)

    # Compute weighted mean along axis 1
    mean1 = ((count3*mean3).sum(1)/count3.sum(1))
    
    return np.where( mean1.mask, nodata, mean1.data)

# Define the burner matrix
# Grid of the input data
x = np.arange(0, 4, 1)
y = np.arange(0, 4, 1)
grid_n, grid_e = np.meshgrid(y,x)

burner = np.zeros((4, 4), dtype=np.dtype([('fre_f',float),('fre_s',float),('residenceTime',float),('burningTime',float),('arrivalTime',float),('grid_e',float),('grid_n',float)])) 
burner = burner.view(np.recarray)
burner.grid_e = grid_e
burner.grid_n = grid_n

burner.fre_f[:,:] = -999
burner.fre_s[:,:] = -999  

burner.fre_f[1:2, 1:2] = 75500
burner.fre_s[1:2, 1:2] = 12750

burner.arrivalTime[1:2, 1:2]= 5
burner.residenceTime[1:2, 1:2]= 10
burner.burningTime[1:2, 1:2]= 25  
##################################
burner.fre_f[1:2, 0:1] = 75500
burner.fre_s[1:2, 0:1] = 12750

burner.arrivalTime[1:2, 0:1]= 5
burner.residenceTime[1:2, 0:1]= 10
burner.burningTime[1:2, 0:1]= 25  
##################################
burner.fre_f[1:2, 2:3] = 50000
burner.fre_s[1:2, 2:3] = 12750

burner.arrivalTime[1:2, 2:3]= 5
burner.residenceTime[1:2, 2:3]= 10
burner.burningTime[1:2, 2:3]= 25    
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

arr = burner.fre_f
mm = downgrade_resolution_4nadir(arr, [2,2], flag_interpolation='sum')

ax = plt.subplot(121)
plt.imshow(arr.T, origin='lower')
ax = plt.subplot(122)
plt.imshow(mm.T, origin='lower')
plt.show()



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


    colors = ["ORANGE", "RED", "BLUE", "GREEN", "YELLOW"]

    # Create the Burner object with the original templates
    template = Burner(nml)


    act_pixels = np.where((burner.fre_f > 0) | (burner.fre_s > 0))

    xmin = grid_e[act_pixels]
    xmax = xmin + 1
    ymin = grid_n[act_pixels]
    ymax = ymin + 1
    #Duration change between times
    v = .1

    w = .5

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

    hrr_act_pix, total_HRR = sum_HRR_per_pixel(burner)
    for burner_id1, hrr_value in hrr_act_pix:
        print(f"Suma de HRR por píxel-> {burner_id1}: {hrr_value:.4f} KJ")
    print(f"Suma total de HRR en todos los píxeles: {total_HRR:.4f} KJ")

