import wget
import logging
import os
import requests
import re
from pathlib import Path

logging.basicConfig(
    filename='sst_download.log',      # Name of the log file
    filemode='w',             # 'a' for appending; use 'w' to overwrite each time
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO        # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
)

for year in range(1981,2024):
    dir = Path(f"D:\\SCI400 - Project\\SCI400-Project-Milks\\sst_1981_2023\\{year}")
    dir_str = f"D:\\SCI400 - Project\\SCI400-Project-Milks\\sst_1981_2023\\{year}"
    if os.path.exists(dir) == False:
        os.makedirs(dir)
    response = requests.get(f"https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/{year}/data/").text
    ssts = re.findall('\d+.*day.*nc"',response)
    for sst in ssts:
        current_files = [avhrr.name for avhrr in dir.iterdir()]
        if sst[:-1] not in current_files:
            if int(sst[4:6]) > 6:
                url = f'https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/{year}/data/{sst[:-1]}'
                wget.download(url,out=dir_str)
            elif (int(sst[4:6]) == 6 and int(sst[6:8]) >= 2) or (int(sst[4:6]) == 12 and int(sst[6:8]) < 25):
                url = f'https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/{year}/data/{sst[:-1]}'
                wget.download(url,out=dir_str)
    logging.info(f'{year} complete.')


# #Retrieve NOAA Pathfinder meteorological data (~15 min/year -> 10.5 hours & 613 GB)
# for year in range(1981,2024):
#     response = requests.get(f"https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/{year}/data/").text
#     ssts = re.findall('\d+.*day.*nc"',response)
#     #dir = os.path.join("D:\SCI400 - Project\SCI400-Project-Milks",f'sst_1981_2023/{year}')
#     dir = f"D:\SCI400 - Project\SCI400-Project-Milks\sst_1981_2023\{year}"
#     if os.path.exists(dir) == False:
#         os.makedirs(dir)
#     else: print("Found")


#     for sst in ssts:
#         url = f'https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/{year}/data/{sst[:-1]}'
#         try:
#             wget.download(url,out=dir)
#             logging.info(f'{sst[0:8]} downloaded.')
#         except Exception as e:
#             logging.error(f'{sst[0:8]} error: {e}')
#     logging.info(f'{year} complete.')

# logging.basicConfig(
#     filename='sst_trimming.log',      # Name of the log file
#     filemode='w',             # 'a' for appending; use 'w' to overwrite each time
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     level=logging.INFO        # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
# )

# #Trim to hurricane season
# root = Path("D:\\SCI400 - Project\\SCI400-Project-Milks\\sst_1981_2023")

# for year in root.iterdir():
#     for avhrr in year.iterdir():
#         if 6 <= int(avhrr.name[4:6]) <= 10:
#             if (int(avhrr.name[4:6]) == 6 and int(avhrr.name[6:8]) < 2):
#                 logging.warning(f"Removing: {avhrr.name[0:8]}")
#                 os.remove(avhrr)
#             else:
#                 logging.info(f"Saving: {avhrr.name[0:8]}")
#         else:
#             logging.warning(f"Removing: {avhrr.name[0:8]}")
#             os.remove(avhrr)
#     logging.info(f"Hurricane active days in {year.name}: {len(os.listdir(year))}")