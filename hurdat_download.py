#HURDAT Parsing -> Hurricane Objects w/ list of Entry objects
from dataclasses import dataclass
from datetime import datetime
import wget
import os
import pandas as pd
import seaborn as sns
import numpy as np

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

@dataclass
class Entry():
    date: datetime
    identifier: str
    status: str
    coordinates: tuple
    max_wind: int
    min_pressure: int
    radius: int

    def __str__(self):
        return f'{self.date} at {self.coordinates}\nID: {self.identifier}, Status: {self.status}\nMax Wind: {self.max_wind}, Min Pressure: {self.min_pressure}, Radius: {self.radius}'

    @classmethod
    def Factory(cls, entry):
        return Entry(date=datetime.strptime(f"{entry[0]}{entry[1]}",'%Y%m%d%H%M'),
                    identifier=entry[2],
                    status=entry[3],
                    coordinates=(entry[4], entry[5]),
                    max_wind=int(entry[6]),
                    min_pressure=int(entry[7]),
                    radius=int(entry[20]))

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
hurricanes_classed=[]
for i in splt_hurricanes_raw:
    hurricanes_classed.append(Hurricane.from_splt_hur(i))

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
latmax = 0
latmin = 180
longmin = 0 # EAST
longmax = 0 # WEST
for hur in hurricanes_classed:
    for entry in hur.entries:
        if float(entry.coordinates[0][:-1]) > latmax:
            max = float(entry.coordinates[0][:-1])
        if float(entry.coordinates[0][:-1]) < latmin:
            min = float(entry.coordinates[0][:-1])
for hur in hurricanes_classed:
    for entry in hur.entries:
        if entry.coordinates[1][-1] == 'W' and float(entry.coordinates[1][:-1]) > longmin:
            longmin = float(entry.coordinates[1][:-1])
        elif entry.coordinates[1][-1] == 'E' and float(entry.coordinates[1][:-1]) > longmax:
            longmax = float(entry.coordinates[1][:-1])
print(max, min) # Range 7.0N - 70.7N
print(longmin,longmax) # Range 136.9W - 13.5E

# Find max & min lat, long hurricane birth
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
print(latmin,latmax) # Range 7.0N - 47.2N
print(longmin,longmax) # Range 97.4W - 16.8W



