import os
from glob import glob
import re
import subprocess
from tkinter import filedialog
from tkinter import *

# Get the mosaic's filepath
# PATH_IN = Tk()
# PATH_IN.withdraw()
# PATH_IN = filedialog.askopenfilename(initialdir = "D:/2_data/1_drone",
#                                      title = "Provide the input tif")

for year in [2020,2021]:
    
    for site in ['TLB','Ridge','CBH']:
        
        for sens in ['TIR','MSP','DSM']:
            
            # 1. CNOFIGURE PATHS AND LOAD DATA
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            
            if sens == 'TIR':
                fn = '*_mosaic_thermal ir.tif'
                
                if site == 'Ridge':
                    fn = '*_index_thermal_ir.tif'
            
            elif sens == 'MSP':
                fn = 'reflectance.tif'
                
            elif sens == 'DSM':
                fn = '*dsm.tif'
            
            PATH_IN = "C:/data/0_Kytalyk/0_drone/{}/{}/{}/original/".format(year, site,sens) 
            
            if not glob(PATH_IN + fn): # Check if list is empty, if yes, continue with next iteration
                print('No data for {} from {} in {}. Skipping...'.format(sens,site,year))
                continue
            
            PATH_IN = glob(PATH_IN + fn)[0]
            # os.chdir(PATH_IN)
            # print(os.curdir)

            fpath, fname = os.path.split(PATH_IN)
            fname = os.path.basename(PATH_IN).split('.')[0]
            
            os.chdir(fpath)
            # print(os.curdir)
            
            """
            m = [re.search(pat,fpath) for pat in ['CBH','TLB','Ridge'] if re.search(pat,fpath) is not None]
            
            site = m[0].group(0)
            print(site)
            """
            
            # Ask for the target resolution and resampling method
            # t_res = input('Type in your target resoultion (in m): \n')
            t_res = str(.15)
            t_res_neg = str(-.15)
            
            methodlist = ('\n').join(['near','bilinear','cubic','cubicspline','lanczos','average'])
            # print(methodlist)
            # method = input('Provide the interpolation method: \n')
            method = 'average'
            
            # Ask for output directory
            # PATH_OUT = Tk()
            # PATH_OUT.withdraw()
            # PATH_OUT = filedialog.askdirectory(initialdir = "D:/2_data/1_drone",
            #                                    title = "Where should I store the resampled TIFFs?")
            PATH_OUT = "C:/data/0_Kytalyk/0_drone/{}/{}/{}/resampled/".format(year, site,sens)
            
            # Concatenate list to form output filename
            fname_out = '_'.join([fname, method,t_res,'resampled.tif'])
            
            if os.path.exists(PATH_OUT+fname_out):
                print('File already exists. Skipping... \n')
                continue
            
            # Define target extent (necessary for grid alignment):  <xmin ymin xmax ymax>
            if site == 'TLB':
                # TLB 2021 proj4
                proj_TIR = '+proj=utm +zone=55 +datum=WGS84 +units=m +no_defs +x_0=517910.084200000041164 +y_0=7859161.502500000409782'
                # proj_MSP = '+proj=utm +zone=55 +datum=WGS84 +units=m +no_defs +x_0=517856.149270000052638 +y_0=7859211.258260000497103'
                xmin = 517910.084
                xmax = 518582.234
                ymin = 7858641.003
                ymax = 7859161.503
                
            elif site == 'Ridge':
                # Ridge 2021 proj4
                proj_TIR = '+proj=utm +zone=55 +datum=WGS84 +units=m +no_defs +x_0=516468.945960000040941 +y_0=7858648.509120000526309'
                # proj_MSP = '+proj=utm +zone=55 +datum=WGS84 +units=m +no_defs +x_0=517856.149270000052638 +y_0=7859211.258260000497103'
                xmin = 516468.945957
                xmax = 517216.743612
                ymin = 7858142.321060
                ymax = 7858648.509123
                
            elif site == 'CBH':
                # CBH 2021 proj4
                proj_TIR = '+proj=utm +zone=55 +datum=WGS84 +units=m +no_defs +x_0=517081.748800000059418 +y_0=7859026.249200000427663'
                xmin = 517081.7488
                xmax = 517794.0988
                ymin = 7858449.7992
                ymax = 7859026.2492
            
            # Run GDALWARP as cmd in subprocess
            print('Resampling {} from {} in {}'.format(sens,site,year))
            args = ['gdalwarp', '-t_srs', proj_TIR, '-r', method, '-te', str(xmin), str(ymin), str(xmax), str(ymax),'-tr', t_res, t_res_neg,   PATH_IN, PATH_OUT + '/' + fname_out]
            subprocess.Popen(args)
        