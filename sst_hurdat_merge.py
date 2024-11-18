from pathlib import Path
import os
import logging

logging.basicConfig(
    filename='sst_trimming.log',      # Name of the log file
    filemode='w',             # 'a' for appending; use 'w' to overwrite each time
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO        # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
)

root = Path("D:\\SCI400 - Project\\SCI400-Project-Milks\\sst_1981_2023")

for year in root.iterdir():
    for avhrr in year.iterdir():
        if 6 <= int(avhrr.name[4:6]) <= 10:
            if (int(avhrr.name[4:6]) == 6 and int(avhrr.name[6:8]) < 2):
                logging.warning(f"Removing: {avhrr.name[0:8]}")
                os.remove(avhrr)
            else:
                logging.info(f"Saving: {avhrr.name[0:8]}")
        else:
            logging.warning(f"Removing: {avhrr.name[0:8]}")
            os.remove(avhrr)
    logging.info(f"Hurricane active days in {year.name}: {len(os.listdir(year))}")

