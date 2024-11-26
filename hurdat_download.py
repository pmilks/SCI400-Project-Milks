#HURDAT Parsing -> Hurricane Objects w/ list of Entry objects
from dataclasses import dataclass
from datetime import datetime
import wget
import os
import pandas as pd
import numpy as np
from data_manip import cdf4_to_pd,find_date
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from pytorch_tabular import TabularModel
from pytorch_tabular.config import (
    DataConfig, TrainerConfig, OptimizerConfig
)
from pytorch_tabular.models import TabTransformerConfig

logging.basicConfig(
    filename='hur_ssts.log',      # Name of the log file
    filemode='w',             # 'a' for appending; use 'w' to overwrite each time
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO        # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
)

COORD_INT = 1


@dataclass
class Hurricane():
    name: str
    entries: list
    year: str
    basin: str
    atfc: int

    def __str__(self):
        return f'Storm {self.name} in {self.year} with {len(self.entries)} entries.'

    @classmethod
    def from_splt_hur(cls,splt_hur):
        splt_hur_head=[i.strip(' ') for i in splt_hur[0].split(',')]
        new = Hurricane(name=splt_hur_head[1],
                        year=int(splt_hur_head[0][-4:]),
                        basin=splt_hur_head[0][:2],
                        atfc=splt_hur_head[0][-6:-4],
                        entries=[])
        new.entries=Entry.entries_factory(splt_hur[1:])
        return new

    def total_max_wind(self):
        max=0
        for entry in self.entries:
            if entry.max_wind > max:
                max = entry.max_wind
        return max
    
    def max_min_pressure(self):
        max=0
        for entry in self.entries:
            if entry.min_pressure > max:
                max = entry.min_pressure
        return max

    def max_radius(self):
        max=0
        for entry in self.entries:
            if entry.radius > max:
                max = entry.radius
        return max
    
    def make_lf(self):
        lfs = []
        for entry in self.entries:
            if entry.identifier == 'L':
                lfs.append(entry)
        if lfs:
            return lfs
        else:
            return False
    
    def lifespan(self):
        return (self.entries[0].date,self.entries[-1].date)

@dataclass
class Entry():
    date: datetime
    identifier: str
    status: str
    coordinates: tuple
    max_wind: int
    min_pressure: int
    radius: int
    sst: float # add temp using date & coords (change coords to negative/positive)
    sst_dev : float
    sea_wind : float

    def __str__(self):
        return f'{self.date} at {self.coordinates}\nID: {self.identifier}, Status: {self.status}\nMax Wind: {self.max_wind}, Min Pressure: {self.min_pressure}, Radius: {self.radius}'

    def convert_coords(self):
        lat = float(self.coordinates[0][:-1])
        if self.coordinates[1][-1] == 'W':
            lon = float(self.coordinates[1][:-1])*-1
        else:
            lon = float(self.coordinates[1][-1])
        return (lat,lon)

    def find_temps(self, coord_int):
        coords = self.convert_coords()
        file = find_date(self.date.strftime("%Y%m%d"))
        avhrr = cdf4_to_pd(file,coord_int)
        sst_sstd = avhrr[(avhrr['Latitude'] == round(coords[0])) & (avhrr['Longitude'] == round(coords[1]))][['SST','SST Deviation']]
        self.sst = round(sst_sstd['SST'].iloc[0],2)
        self.sst_dev = round(sst_sstd['SST Deviation'].iloc[0],2)

    def find_wind(self,coord_int):    
        coords = self.convert_coords()
        file = find_date(self.date.strftime("%Y%m%d"))
        avhrr = cdf4_to_pd(file,coord_int)
        return avhrr[(avhrr['Latitude'] == round(coords[0])) & (avhrr['Longitude'] == round(coords[1]))]['Wind Speed'].iloc[0]

    @classmethod
    def Factory(cls, entry):
        return Entry(date=datetime.strptime(f"{entry[0]}{entry[1]}",'%Y%m%d%H%M'),
                    identifier=entry[2],
                    status=entry[3],
                    coordinates=(entry[4], entry[5]),
                    max_wind=int(entry[6]),
                    min_pressure=int(entry[7]),
                    radius=int(entry[20]),
                    sst=0,
                    sst_dev=0,
                    sea_wind=0)

    @classmethod
    def entries_factory(cls,raw_entries):
        entries=[]
        for entry in raw_entries:
            entry_splt=[i.strip(' ') for i in entry.split(',')]
            entries.append(Entry.Factory(entry_splt))
        return entries

if os.path.exists("hurdat2-1851-2023-051124.txt") == False:
    wget.download("https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt")

if os.path.exists("hurdat2-1981-2023.txt") == False:
    with open("hurdat2-1851-2023-051124.txt", "r") as hurdat_1851:
        with open("hurdat2-1981-2023.txt", "w") as hurdat_1981:
            found_1981 = False
            for line in hurdat_1851:
                if line[0:8] == 'AL121981':
                    found_1981 = True
                if found_1981:
                    hurdat_1981.write(line)

# Seperate individual hurriances with their entries
# Output : list of str
def txt_to_hur_classed():
    splt_hurricanes_raw=[]
    with open("hurdat2-1981-2023.txt","r") as f:
        hur=[]
        for line in f:
            if line[0].isalpha():
                if hur != []:
                    splt_hurricanes_raw.append(hur)
                    hur=[]
            hur.append(line.strip())

    # Str hurricanes into Hurricane classes
    hurricanes_classed = []
    for i in splt_hurricanes_raw:
        new_hur = Hurricane.from_splt_hur(i)
        # for entry in new_hur.entries:
        #     entry.find_temps()
        hurricanes_classed.append(new_hur)
    return np.array(hurricanes_classed)

# hurricanes_classed = txt_to_hur_classed()

# df = pd.DataFrame(columns=['ID','Lat','Lon','SST','SSTd','hurMaxWind','hurMinPressure','hurRadius',
#                             'Next Lat','Next Lon','Next hurMaxWind','Next hurMinPressure','Next hurRadius']) # NO SEA WIND
# df.to_csv('traj_input.csv',index=False)

# # Model input
def class_to_pd_csv(coord_int):
    # Class to pd
    for hur in hurricanes_classed[-15:]:
        df = pd.DataFrame(columns=['ID','Lat','Lon','SST','SSTd','hurMaxWind','hurMinPressure','hurRadius',
                                'Next Lat','Next Lon','Next hurMaxWind','Next hurMinPressure','Next hurRadius']) # NO SEA WIND
        current_entry = False
        next_entry = False
        first_landfall = False
        for entry in hur.entries:
            if first_landfall == False:
                coords = entry.convert_coords()
                try:
                    entry.find_temps(coord_int)
                    if (entry.max_wind < 0 or entry.min_pressure < 0 or entry.radius < 0) == True:
                        current_entry = False
                    else:
                        if current_entry == False:
                            current_entry = [f"{hur.basin}{hur.atfc}{hur.year} - {entry.date.strftime('%m%d %H')}h",coords[0],coords[1],entry.sst,entry.sst_dev,entry.max_wind,entry.min_pressure,entry.radius]
                        else:
                            next_entry = [coords[0],coords[1],entry.max_wind,entry.min_pressure,entry.radius]
                except IndexError:
                    first_landfall = True
                    if current_entry != False:
                        next_entry = [coords[0],coords[1],entry.max_wind,entry.min_pressure,entry.radius]
                finally:
                    if next_entry != False:
                        df.loc[len(df)] = current_entry + next_entry
                        current_entry = [f"{hur.basin}{hur.atfc}{hur.year} - {entry.date.strftime('%m%d %H')}h",coords[0],coords[1],entry.sst,entry.sst_dev,entry.max_wind,entry.min_pressure,entry.radius]
                        next_entry = False
        df.to_csv('traj_input.csv',mode='a',index=False,header=False)

# df = pd.read_csv('traj_input.csv')
# X = df[['Lat','Lon','SST','SSTd','hurMaxWind','hurMinPressure','hurRadius']]
# y = df[['Next Lat','Next Lon','Next hurMaxWind','Next hurMinPressure','Next hurRadius']]

# model = TabularModel(data_config=DataConfig(
#     target=['Next Lat','Next Lon','Next hurMaxWind','Next hurMinPressure','Next hurRadius'],
#     continuous_cols=['Lat','Lon','SST','SSTd','hurMaxWind','hurMinPressure','hurRadius']
# ),
#     model_config=TabTransformerConfig(
#         task='regression',
#         input_embed_dim=32,
#         num_heads=4,
#         num_attn_blocks=2
# ),
#     optimizer_config=OptimizerConfig(),
#     trainer_config=TrainerConfig(max_epochs=10))

# model.fit(train=df.drop(columns='ID'))
# prediction = model.predict(X)
# prediction.columns = ['Lat','Lon','hurMaxWind','hurMinPressure','hurRadius']
# next_step = X[['Lat','Lon','hurMaxWind','hurMinPressure','hurRadius']] + prediction

# y.columns = ['Lat','Lon','hurMaxWind','hurMinPressure','hurRadius']

# # Evaluate performance
# mse = mean_squared_error(y['Lat'], next_step['Lat'])
# print(f"Mean Squared Error: {mse:.2f}")

# Save the model
# model.save_model("tab_transformer_model")

# Load the model
# loaded_model = TabularModel.load_from_checkpoint("tab_transformer_model")



# hurricanes_classed = txt_to_hur_classed()

# COORD_INT = 5

# df = pd.DataFrame(columns=['Date','Latitude','Longitude','SST','SST Deviation','Birth'])
# df.to_csv('birth_input.csv',mode='w',index=False)

# for hur in hurricanes_classed[-30:]:
#     birth_entry = hur.entries[0]
#     if (birth_entry.date.month > 6 or (birth_entry.date.month == 6 and birth_entry.date.day >= 2)) and (6 <= hur.entries[-1].date.month <= 12):
#         date = birth_entry.date.strftime('%Y%m%d')
#         coords = list(birth_entry.convert_coords())
#         coords[0] = min(range(7,47,COORD_INT), key=lambda x:abs(x-coords[0]))
#         coords[1] = min(range(-97,-17,COORD_INT), key=lambda x:abs(x-coords[1]))
#         if date not in set(df['Date'].values):
#             avhrr = cdf4_to_pd(find_date(date),COORD_INT)
#             avhrr[['Date','Birth']] = date,0
#             avhrr = avhrr[['Date','Latitude','Longitude','SST','SST Deviation','Birth']]
#             df = pd.concat([df,avhrr], ignore_index=True)
#         df.loc[(df['Date'] == date) & (df['Latitude'] == coords[0]) & (df['Longitude'] == coords[1]), 'Birth'] = 1
# df.to_csv('birth_input.csv',mode='a',index=False,header=False)

df = pd.read_csv('birth_input.csv')

X = df[['Latitude','Longitude','SST','SST Deviation']]
y = df['Birth']

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# model = TabularModel(data_config=DataConfig(
#     target=['Next Lat','Next Lon','Next hurMaxWind','Next hurMinPressure','Next hurRadius'],
#     continuous_cols=['Lat','Lon','SST','SSTd','hurMaxWind','hurMinPressure','hurRadius']
# ),
#     model_config=TabTransformerConfig(
#         task='regression',
#         input_embed_dim=32,
#         num_heads=4,
#         num_attn_blocks=2
# ),
#     optimizer_config=OptimizerConfig(),
#     trainer_config=TrainerConfig(max_epochs=10))

model = TabularModel(data_config=DataConfig(
    target='Births',
    continuous_cols=['Latitude','Longitude','SST','SST Deviation'],
),
    model_config=TabTransformerConfig(
        task='classification'
))

model.fit(train=train_data)
result = model.evaluate(test=test_data)
print("Eval: ",result)

predictions = model.predict(test_data)
print("Predictions: ",predictions)
























# Classify Hurriance classes into categories
# hur_cat = []
# for i in hurricanes_classed:
#     hur_cat_ind = [f"{i.name}{i.year}"]
#     hur_cat_ind.append(i.entries[0].coordinates[0])
#     hur_cat_ind.append(i.entries[0].coordinates[1])
#     if i.make_lf():
#         hur_cat_ind.append(1)
#     else:
#         hur_cat_ind.append(0)
#     if i.total_max_wind() >= 96 and i.total_max_wind() <= 112:
#         hur_cat_ind.append(1)
#     else:
#         hur_cat_ind.append(0)
#     if i.total_max_wind() >= 113 and i.total_max_wind() <= 136:
#         hur_cat_ind.append(1)
#     else:
#         hur_cat_ind.append(0)
#     if i.total_max_wind() >= 137:
#         hur_cat_ind.append(1)
#     else:
#         hur_cat_ind.append(0)
#     hur_cat.append(hur_cat_ind)

# Find max & min lat, long hurricane activity
def hur_activity_coords(hurricanes_Classed):
    latmax = 0
    latmin = 180
    longmin = 0 # WEST
    longmax = 0 # EAST
    for hur in hurricanes_classed:
        for entry in hur.entries:
            if float(entry.coordinates[0][:-1]) > latmax:
                latmax = float(entry.coordinates[0][:-1])
            if float(entry.coordinates[0][:-1]) < latmin:
                latmin = float(entry.coordinates[0][:-1])
    for hur in hurricanes_classed:
        for entry in hur.entries:
            if entry.coordinates[1][-1] == 'W' and float(entry.coordinates[1][:-1]) > longmin:
                longmin = float(entry.coordinates[1][:-1])
            elif entry.coordinates[1][-1] == 'E' and float(entry.coordinates[1][:-1]) > longmax:
                longmax = float(entry.coordinates[1][:-1])
    return ((latmax, latmin),(longmin,longmax)) # Lat Range 7.0N - 70.7N, Long Range 136.9W - 13.5E

# Find max & min lat, long hurricane birth
def hur_birth_coords(hurricanes_classed):
    latmax = 0
    longmax = 0
    latmin = 180
    longmin = 180
    for hur in hurricanes_classed:
        if float(hur.entries[0].coordinates[0][:-1]) > latmax:
            latmax = float(hur.entries[0].coordinates[0][:-1])
        if float(hur.entries[0].coordinates[0][:-1]) < latmin:
            latmin = float(hur.entries[0].coordinates[0][:-1])
        if float(hur.entries[0].coordinates[1][:-1]) < longmin:
            longmin = float(hur.entries[0].coordinates[1][:-1])
        elif float(hur.entries[0].coordinates[1][:-1]) > longmax:
            longmax = float(hur.entries[0].coordinates[1][:-1])
    return((latmin,latmax),(longmin,longmax)) # Lat Range 7.0N - 47.2N, Long Range 97.4W - 16.8W


# Hurricane season
def hur_szn_bounds(hurricanes_classed):
    mindate = datetime(1900,12,31)
    maxdate = datetime(1900,1,1)
    for hur in hurricanes_classed:
        if hur.entries[0].date.month < mindate.month and hur.entries[0].date.day < mindate.day:
            mindate = mindate.replace(month=hur.entries[0].date.month)
            mindate = mindate.replace(day=hur.entries[0].date.day)
        if (hur.entries[-1].date.month > maxdate.month) or (hur.entries[-1].date.month == maxdate.month and hur.entries[-1].date.day > maxdate.day):
            maxdate = maxdate.replace(month=hur.entries[-1].date.month)
            maxdate = maxdate.replace(day=hur.entries[-1].date.day)
    return(mindate,maxdate) # Range 06/02 - 10/31


