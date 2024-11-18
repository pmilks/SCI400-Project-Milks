import wget
import logging
import os
import requests
import re

logging.basicConfig(
    filename='sst_download.log',      # Name of the log file
    filemode='w',             # 'a' for appending; use 'w' to overwrite each time
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO        # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
)


if os.path.exists('D:\SCI400 - Project\SCI400-Project-Milks\sst_1981_2023') == False:
    os.makedirs('D:\SCI400 - Project\SCI400-Project-Milks\sst_1981_2023')

#Retrieve NOAA Pathfinder meteorological data (~15 min/year -> 10.5 hours & 613 GB)
for year in range(2003,2024):
    response = requests.get(f"https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/{year}/data/").text
    ssts = re.findall('\d+.*day.*nc"',response)
    #dir = os.path.join("D:\SCI400 - Project\SCI400-Project-Milks",f'sst_1981_2023/{year}')
    dir = f"D:\SCI400 - Project\SCI400-Project-Milks\sst_1981_2023\{year}"
    if os.path.exists(dir) == False:
        os.makedirs(dir)
    else: print("Found")


    for sst in ssts:
        url = f'https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/{year}/data/{sst[:-1]}'
        try:
            wget.download(url,out=dir)
            logging.info(f'{sst[0:8]} downloaded.')
        except Exception as e:
            logging.error(f'{sst[0:8]} error: {e}')
    logging.info(f'{year} complete.')