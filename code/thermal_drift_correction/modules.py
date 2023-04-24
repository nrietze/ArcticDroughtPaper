from glob import glob
import os

import pyexiv2
from PIL import Image
from skimage import io

import datetime as dt
import pandas as pd
import numpy as np
from tqdm import tqdm

def ExtractImageData(filename: str):
    """
    filename: A string containing path + filename of the image
    
    returns: the GPS time of acquisition, sensor temperature, average image values (DN or LST), and filename
    """
    # Load image EXIF and XMP metadata:
    img = pyexiv2.Image(filename,encoding='utf-8')
    exif = img.read_exif()
    data = img.read_xmp()
    img.close()

    # Read and format image acquisition time:
    gps_date = exif['Exif.GPSInfo.GPSDateStamp']
    year, month, day = map(int, gps_date.split(':'))
    gps_time = exif['Exif.GPSInfo.GPSTimeStamp']
    hrs, mins, secs, _ = map(int, gps_time.split('/1'))  # remove all /1 from string ('1/1 26/1 57256/1000')
    secs, ms = map(int, str(secs / 1e3).split(".", 1))
    timestamp = dt.datetime(year, month, day, hrs, mins, secs)
    pd_timestamp = pd.to_datetime(timestamp,format = '%Y-%M-%D %H:%m:%s',utc=True)
    
    lat_raw = exif['Exif.GPSInfo.GPSLatitude']
    deg,mm,ss,_ = map(int, str(lat_raw).split("/1"))
    lat_dd = deg + mm/60 + ss/36e8 # 6e6 bc. /3600/1e6

    lon_raw = exif['Exif.GPSInfo.GPSLongitude']
    deg,mm,ss,_ = map(int, str(lon_raw).split("/1"))
    lon_dd = deg + mm/60 + ss/36e8 # 6e6 bc. /3600/1e6
    
    # Get sensor temperature:
    T_sensor = float(data['Xmp.Camera.SensorTemperature'])
    
    # Compute average digital number (for JPEGs) or average LST (for TIFs)
    img = io.imread(filename)
    
#     if filename.endswith('.jpg'):
#         masked = np.ma.masked_equal(img[:,:,0], 0)
#     elif filename.endswith('.tif'):
#         masked = np.ma.masked_equal(img, 0)
    
    img_mean = img.mean()
    
    return pd_timestamp, T_sensor, img_mean, filename, lat_dd, lon_dd

# =====================================================================

def GetFlightlines(df):
    # 1 = up, 0 = down
    df['direction'] = 0
    flightline = 0
    df['flightline'] = flightline
    i=1
    
    for lat in df['latitude'].iloc[1:]:
        # Up if Latitude increases
        if lat < df['latitude'][i-1]:
            df.loc[i,'direction'] = 0

        #Down if latitude decreases
        else:
            df.loc[i,'direction'] = 1

        # Next flightline if change in direction
        if (df.loc[i,'direction'] != df.loc[i-1,'direction']):
            flightline+=1

        df.loc[i,'flightline'] = flightline

        i+=1
        
    return df

# =====================================================================

def CompileMetadata(filenames: list, SaveToCSV = False):
    """
    returns a Dataframe with:
            - gps_time,
            - T_sensor,
            - mean_tile_LST,
            - filename,
            - latitude,
            - longitude,
            - flighttime_sec,
            - direction,
            - flightline
    """
    df = pd.DataFrame(data = map(ExtractImageData,tqdm(filenames)),
                     columns = ['gps_time', 'T_sensor','mean_tile_LST','filename','latitude','longitude'])
    
    df = GetFlightlines(df)
    
    delta_secs = [(x - df.gps_time[0]).total_seconds() for x in df.gps_time]
    df['flighttime_sec'] = delta_secs
    
    # Get site name
    s = os.path.dirname(filenames[0])
    for ss in ['Ridge','TLB','CBH']:
        try:
            s_ix = s.rfind(ss)
            site = s[s_ix : (s_ix+len(ss))] 
        except:
            print('Cannot decipher the site, please specify.')
            site = input('Provide the name of the site:')
    
    if SaveToCSV:
        filepath = os.path.dirname(filenames[0])
#        filepath = 'C:/data/0_Kytalyk/0_drone/internal_data/'
        df.to_csv(filepath + "/" + 'metadata.csv', sep = ';')
        
    return df

# =====================================================================

def GatherData(df,unc_instr, FitOnUnstable = True):
    df['site'] = df.filename.apply(lambda s: s.replace('\\','/').split('/')[4].split('_')[0])
    df['year'] = df.gps_time.dt.year
    df['flighttime_min'] = df.flighttime_sec.div(60)
    
    # Instrument T uncertainty
    stable_sensorT = df.T_sensor.min()
    df['isStable'] = np.logical_and(df.T_sensor <= stable_sensorT + unc_instr,
                                    df.T_sensor >= stable_sensorT - unc_instr)

    # Compute LST deviation from when sensor is stable
    stable_LST = df.mean_tile_LST[df['isStable']].mean()
    df['LST_deviation'] = df.mean_tile_LST.values - stable_LST
    
    for deg in range(1,4):
        
        if FitOnUnstable:
#             Fits the correction model only on the unstable data
            z = PolyFitSensorT(df[~df['isStable']],xvar = 'T_sensor', yvar = 'LST_deviation', degree = deg)
            df['fit_%i'% deg] = z(df.T_sensor)
        
        else:
#             Fits the model on all tiles
            z = PolyFitSensorT(df,xvar = 'T_sensor', yvar = 'LST_deviation', degree = deg)
            df['fit_%i'% deg] = z(df.T_sensor)
        
        if (df.site.unique() == 'TLB') or (df.site.unique() == 'Ridge') & (df.year.unique() == 2020):
            cutoff = 13
            # Mae 2 fits for 2020 TLB
            cond1 = df.flighttime_min <= cutoff
            z1 = PolyFitSensorT(df[cond1],
                                xvar = 'T_sensor', 
                                yvar = 'LST_deviation', 
                                degree = deg)
            df.loc[cond1,'fit_%i'% deg] = z1(df[cond1].T_sensor)
            
            cond2 = df.flighttime_min > cutoff
            z2 = PolyFitSensorT(df[cond2],
                                xvar = 'T_sensor', 
                                yvar = 'LST_deviation', 
                                degree = deg)
            df.loc[cond2,'fit_%i'% deg] = z2(df[cond2].T_sensor)
            
    return df

# =====================================================================

def PolyFitSensorT(df: pd.DataFrame, xvar:str, yvar:str, degree: int):
    """
    df: Dataframe that contains the dependent and explanatory variable (e.g. sensor Temperature and average tile LST)
    
    returns a numpy polynomial
    """
    x = df[xvar]
    y = df[yvar]
    
    return np.poly1d(np.polyfit(x,y,degree))

# =====================================================================

def ExportTIFF(data: np.array, filename: str, filename_out:str):
    """
    data: 2-D numpy array of image data
    filename: path + filename of original image
    filename_out: path + name of TIFF to save
    """
    # Write image data to TIFF
    im = Image.fromarray(data)
    im.save(filename_out)
    
    # Read metadata from original file
    img = pyexiv2.Image(filename)
    exif = img.read_exif()
    data = img.read_xmp()
    img.close()

    # Write metadata from original file to new image file
    img_new = pyexiv2.Image(filename_out)
    img_new.modify_exif(exif)
    img_new.modify_xmp(data)
    img_new.close()
    
# =====================================================================
    
def IO_correct_tif(filename: str,filename_out: str, fit: str , df: pd.DataFrame):
    """
    This function subtracts the fitted LST from all pixels in that tile using the sensor temperature during that acquisition.

    filename: path + name of the target image
    df: Dataframe that contains the dependent and explanatory variable (e.g. sensor Temperature and average tile LST)
    fit: a string indicating the polynomial fitted column name
        (e.g., fit_1 linear, fit_2 quadratic, and fit_3 for cubic)
    """
    I = io.imread(filename)
    img_gps_time = df.index[df.filename.str.contains(os.path.basename(filename))]
    # img_sensorT = df.T_sensor.loc[img_gps_time]
    # II = I - fit(img_sensorT)
    
    correction_factor = df.loc[img_gps_time,fit].item()
    
    II = I - correction_factor
    
    if filename.endswith('.jpg'):
        print('Saving to JPEG not implemented!')
        pass
    elif filename.endswith('.tif'):
        ExportTIFF(data = II, filename = filename, filename_out = filename_out)
        
        
    return II
