from glob import glob
import os
from tkinter import filedialog
from tkinter import *

from tqdm import tqdm

from skimage import io

import pandas as pd
import numpy as np

from modules import *
# =====================================================================

# Search for all 
TIF_PATHS = Tk()
TIF_PATHS.withdraw()
TIF_PATHS = filedialog.askdirectory(title = "Where are the single tile thermal TIFFs?")
TIF_PATHS = [TIF_PATHS + '/']

for PATH_IN in TIF_PATHS:
    r = "y"
    print(PATH_IN)
    # Check if target directory has drift-corrected scenes
    print("Processing %s ... \n" % PATH_IN)
    
    # Locate all image tiles in the input folder
    flist = sorted(glob(PATH_IN + '*.tif'))

    # Check existing data & extract metadata :
    # °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°
    # No metadata table found --> compile it        
    if not os.path.exists(PATH_IN + 'metadata.csv'):
        print("Compiling metadata...")
        df = CompileMetadata(flist, SaveToCSV = True)

    # Metadata found --> how to proceed
    elif os.path.exists(PATH_IN + 'metadata.csv'):
        # Ask user if the site needs to be processed again
        r = input("You might have already processed this flight. Continue with old metadata? \n y/n \n")
        if r == "y":
            print("Loading existing metadata...")
            df = pd.read_csv(PATH_IN + 'metadata.csv', header = 0, sep = ';')
        elif r == "n":
            print("Compiling metadata...")
            df = CompileMetadata(flist, SaveToCSV = True)
    
    # Instrument T uncertainty
    unc_instr = .1 # +/- 0.1 °C deviation from temperature is unstable
    
    df = GatherData(df,
                    unc_instr, FitOnUnstable = True)
    
    # Correct and store all images in that iteration to a new directory
    PATH_OUT = Tk()
    PATH_OUT.withdraw()
    PATH_OUT = filedialog.askdirectory(title = "Where should I store the corrected TIFFs?")
    
#     NAMES = [PATH_OUT + os.path.basename(fn) for fn in flist]
    print("Applying drift correction and saving to TIFF")
    
    [IO_correct_tif(filename = fn ,
                    filename_out = PATH_OUT +  "/" + os.path.basename(fn), 
                    fit = 'fit_2' , df = df) for fn in tqdm(flist)]
    print("done.")
    
    

