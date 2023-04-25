import os
from glob import glob
import subprocess

fpath = "../../../data/mosaics/"
os.chdir(fpath)

for year in [2020,2021]:
    
    for site in ['TLB','Ridge','CBH']:
        
        for sens in ['thermal','msp']:
            
            # 1. CNOFIGURE PATHS AND LOAD DATA
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            fname = f"{site}_{sens}_{year}.tif"
            
            if not glob(fname): # Check if list is empty, if yes, continue with next iteration
                print(f'No {sens} data from {site} in {year}. Skipping...')
                continue
            
            # Ask for the target resolution and resampling method
            t_res = str(.15)
            t_res_neg = str(-.15)
            
            # list of possible resampling methods, we used average
            methodlist = ('\n').join(['near','bilinear','cubic','cubicspline','lanczos','average'])
            method = 'average'
            
            # Concatenate list to form output filename
            fname_out = f"{site}_{sens}_{year}_resampled.tif"
            
            if os.path.exists(fname_out):
                print('File already exists. Skipping... \n')
                continue
            
            # Define target extent (necessary for grid alignment):  <x_min y_min x_max y_max>
            if site == 'TLB':
                # TLB 2021 proj4
                proj_TIR = '+proj=utm +zone=55 +datum=WGS84 +units=m +no_defs +x_0=517910.084200000041164 +y_0=7859161.502500000409782'
                x_min = 517910.084
                x_max = 518582.234
                y_min = 7858641.003
                y_max = 7859161.503
                
            elif site == 'Ridge':
                # Ridge 2021 proj4
                proj_TIR = '+proj=utm +zone=55 +datum=WGS84 +units=m +no_defs +x_0=516468.945960000040941 +y_0=7858648.509120000526309'
                x_min = 516468.945957
                x_max = 517216.743612
                y_min = 7858142.321060
                y_max = 7858648.509123
                
            elif site == 'CBH':
                # CBH 2021 proj4
                proj_TIR = '+proj=utm +zone=55 +datum=WGS84 +units=m +no_defs +x_0=517081.748800000059418 +y_0=7859026.249200000427663'
                x_min = 517081.7488
                x_max = 517794.0988
                y_min = 7858449.7992
                y_max = 7859026.2492
            
            # Run GDALWARP as cmd in subprocess
            print(f'Resampling {sens} from {site} in {year}')
            args = ['gdalwarp', 
                    '-t_srs', proj_TIR, 
                    '-r', method, 
                    '-te', str(x_min), str(y_min), str(x_max), str(y_max),
                    '-tr', t_res, t_res_neg,
                    fname, fname_out]
            subprocess.Popen(args)
        