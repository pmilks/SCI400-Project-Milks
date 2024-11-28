#HURDAT Parsing -> Hurricane Objects w/ list of Entry objects
from dataclasses import dataclass
from datetime import datetime, timedelta
import wget
import os
import pandas as pd
import numpy as np
from data_manip import cdf4_to_pd,find_date
import logging
import torch
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from pytorch_tabular import TabularModel
from pytorch_tabular.config import (
    DataConfig, TrainerConfig, OptimizerConfig, ModelConfig
)
from pytorch_tabular.models import TabTransformerConfig
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(
    filename='hur_ssts.log',      # Name of the log file
    filemode='w',             # 'a' for appending; use 'w' to overwrite each time
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO        # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
)

COORD_INT = 1
WIND_DEATH = 31.31
PRESSURE_DEATH = 1002.82
RADIUS_DEATH = 67.89

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

    def deconvert_coords(coords):
        lat = f'{str(coords[0])}N'
        if coords[1] < 0:
            lon = f'{str(coords[1]*-1)}W'
        else:
            lon = f'{str(coords[1])}E'
        return (lat,lon)

    def find_temps(self, coord_int):
        coords = self.convert_coords()
        file = find_date(self.date.strftime("%Y%m%d"))
        if file == -1:
            # print(self.date.strftime("%Y%m%d"))
            raise FileNotFoundError(f'{self.date.strftime("%Y%m%d")}]')
        avhrr = cdf4_to_pd(file,coord_int)
        sst_sstd = avhrr[(avhrr['Latitude'] == round(coords[0])) & (avhrr['Longitude'] == round(coords[1]))][['SST','SST Deviation']]
        self.sst = round(sst_sstd['SST'].iloc[0],2)
        self.sst_dev = round(sst_sstd['SST Deviation'].iloc[0],2)

    def find_wind(self,coord_int):    
        coords = self.convert_coords()
        file = find_date(self.date.strftime("%Y%m%d"))
        avhrr = cdf4_to_pd(file,coord_int)
        return avhrr[(avhrr['Latitude'] == round(coords[0])) & (avhrr['Longitude'] == round(coords[1]))]['Wind Speed'].iloc[0]

    def to_list(self):
        return[self.date, self.identifier, self.status,self.coordinates,self.max_wind,
               self.min_pressure, self.radius, self.sst, self.sst_dev, self.sea_wind]

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

# hur_2021 = []
# for hur in hurricanes_classed:
#     if hur.entries[0].date.year >= 2021:
#         hur_2021.append(hur)

# df = pd.DataFrame(columns=['ID','Lat','Lon','SST','SSTd','hurMaxWind','hurMinPressure','hurRadius',
#                             'Next Lat','Next Lon','Next hurMaxWind','Next hurMinPressure','Next hurRadius']) # NO SEA WIND
# df.to_csv('traj_input.csv',index=False)

# Model input
def class_to_traj_csv(coord_int):
    for hur in hur_2021:
        temp_df = pd.DataFrame(columns=['ID','Lat','Lon','SST','SSTd','hurMaxWind','hurMinPressure','hurRadius',
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
                        print(entry.max_wind,entry.min_pressure,entry.radius)
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
                except Exception:
                    current_entry = False
                    continue
                finally:
                    if next_entry != False:
                        temp_df.loc[len(temp_df)] = current_entry + next_entry
                        current_entry = [f"{hur.basin}{hur.atfc}{hur.year} - {entry.date.strftime('%m%d %H')}h",coords[0],coords[1],entry.sst,entry.sst_dev,entry.max_wind,entry.min_pressure,entry.radius]
                        next_entry = False
        temp_df.to_csv('traj_input.csv',mode='a',index=False,header=False)

# class_to_traj_csv(COORD_INT)

def train_trajectory_trans(df,base_entry:list):
    continuous_features = ['Lat','Lon','SST','SSTd','hurMaxWind','hurMinPressure','hurRadius']
    target = ['Next Lat','Next Lon','Next hurMaxWind','Next hurMinPressure','Next hurRadius']

    X_cont = df[continuous_features].values
    y = df[target].values

    scaler_X = StandardScaler()
    X_cont = scaler_X.fit_transform(X_cont) # TEST DATA MUST BE SCALED X_test_scaled = scaler_X.transform(X_test) since both pre and post scaled

    scalers_y = [StandardScaler() for _ in range(y.shape[1])]
    y = np.hstack([scaler.fit_transform(y[:, i].reshape(-1, 1)) for i, scaler in enumerate(scalers_y)])

    if os.path.exists("trajectory_tf.pth") == False:
        X_cont = torch.tensor(X_cont, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        model = TabTransformer(      
            categories=[],                               # No categorical features
            num_continuous=len(continuous_features),     # Number of continuous features
            dim=32,                                      # Embedding dimension (paper suggests 32)
            dim_out=y.shape[1],                          # Output dimension equals the number of target columns
            depth=6,                                     # Depth of the transformer (paper recommends 6)
            heads=8,                                     # Number of attention heads (paper suggests 8)
            attn_dropout=0.1,                            # Attention dropout rate
            ff_dropout=0.1,                              # Feed-forward dropout rate
        )

        class HurricaneDataset(Dataset):
            def __init__(self, X_cont, y):
                self.X_cont = X_cont
                self.y = y

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return self.X_cont[idx], self.y[idx]

        X_cont_train, X_cont_val, y_train, y_val = train_test_split(
            X_cont, y, test_size=0.2, random_state=42
        )

        train_dataset = HurricaneDataset(X_cont_train, y_train)
        val_dataset = HurricaneDataset(X_cont_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        criterion = nn.MSELoss()  # Use Mean Squared Error Loss for regression
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        epochs = 20
        # val_loss_dict = {}
        for epoch in range(epochs):
            model.train()
            for X_cont_batch, y_batch in train_loader:
                optimizer.zero_grad()
                dummy_cat = torch.empty((X_cont_batch.size(0), 0), dtype=torch.long)
                outputs = model(dummy_cat, X_cont_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_cont_batch, y_batch in val_loader:
                    dummy_cat = torch.empty((X_cont_batch.size(0), 0), dtype=torch.long)
                    outputs = model(dummy_cat, X_cont_batch)
                    val_loss += criterion(outputs, y_batch).item()

            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(val_loader):.4f}")
            # val_loss_dict[epoch] = val_loss/len(val_loader)

        # plt.figure(figsize=(10, 6))
        # plt.plot(val_loss_dict.keys(), val_loss_dict.values(), label="Validation Loss", marker='o')
        # plt.title("Trajectory Transformer Validation Loss")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        torch.save(model, "trajectory_tf.pth")
        print("Model saved successfully at trajectory_transf.pth!")

    if base_entry == None:
        return

    traj_model = torch.load("trajectory_tf.pth")

    # Example predictions with the trained model
    traj_model.eval()
    with torch.no_grad():
        sample_cont = torch.tensor(scaler_X.transform([base_entry]), dtype=torch.float)  # Example continuous feature values
        dummy_cat = torch.empty((sample_cont.size(0), 0), dtype=torch.long)  # Empty categorical input
        prediction = traj_model(dummy_cat, sample_cont)
        y_pred_original = np.hstack([scaler.inverse_transform(prediction[:, i].reshape(-1, 1)) for i, scaler in enumerate(scalers_y)])
        return y_pred_original[0]

# ex_entry = Entry(datetime(2023,8,30,0,0,0), 'pAL122023',0,('14.9N','21.3W'),25,1008,90,275.83,-0.8,0)

def is_storm_alive(max_wind, min_pressure, radius):
    check = 0
    if max_wind < 31.32:
        check += 1
    if min_pressure < 1002.82:
        check += 1
    if radius < 67.89:
        check += 1
    if check >= 2:
        return False
    return True

#COMPLETE, GROW DATA
def trajectory_trans_pred(df,base_entry:Entry): # 'Date','Lat','Lon','SST','SSTd','hurMaxWind','hurMinPressure','hurRadius' -> 'Next Lat','Next Lon','Next hurMaxWind','Next hurMinPressure','Next hurRadius'
    traj_df = pd.DataFrame(columns=['Date','Lat','Lon','SST','SSTd','hurMaxWind','hurMinPressure','hurRadius','Status'])
    traj_df.loc[len(traj_df)] = [base_entry.date, base_entry.convert_coords()[0], base_entry.convert_coords()[1],
                                 base_entry.sst, base_entry.sst_dev, base_entry.max_wind, base_entry.min_pressure,
                                 base_entry.radius, base_entry.status]
    dead_storm = False
    while(dead_storm == False):
        next_date = base_entry.date + timedelta(hours=6)
        coords = base_entry.convert_coords()
        next_traj = train_trajectory_trans(df, [coords[0],coords[1],base_entry.sst,base_entry.sst_dev,
                                                base_entry.max_wind, base_entry.min_pressure, base_entry.radius])
        new_entry = Entry(date=next_date, identifier=base_entry.identifier, status='A',
                        coordinates=Entry.deconvert_coords(next_traj[0:2]), max_wind=next_traj[2], 
                        min_pressure=next_traj[3], radius=next_traj[4], sst=0, sst_dev=0, sea_wind=0)
        if len(traj_df) > 4:
            if is_storm_alive(new_entry.max_wind,new_entry.min_pressure,new_entry.radius) == False:
                new_entry.status = 'D'
                dead_storm = True
        try:
            new_entry.find_temps(COORD_INT)
        except IndexError:
            new_entry.status = 'L'
            dead_storm = True
        except FileNotFoundError:
            return ('Missing day')
        finally:
            traj_df.loc[len(traj_df)] = [new_entry.date, next_traj[0], next_traj[1],
                                         new_entry.sst, new_entry.sst_dev, new_entry.max_wind, new_entry.min_pressure,
                                         new_entry.radius, new_entry.status]
            base_entry = new_entry
        if len(traj_df) > 9:
            return traj_df
    return traj_df

df = pd.read_csv('traj_input.csv')
train_trajectory_trans(df,None)

COORD_INT = 1

# df = pd.DataFrame(columns=['Date','Latitude','Longitude','SST','SST Deviation',
#                            'Max Wind','Min Pressure','Radius','Birth'])
# df.to_csv('birth_input.csv',mode='w',index=False)

# for hur in hur_2021:
#     birth_entry = hur.entries[0]
#     if (birth_entry.date.month > 6 or (birth_entry.date.month == 6 and birth_entry.date.day >= 2)) and (6 <= hur.entries[-1].date.month <= 12):
#         date = birth_entry.date.strftime('%Y%m%d')
#         coords = list(birth_entry.convert_coords())
#         coords[0] = min(range(7,47,COORD_INT), key=lambda x:abs(x-coords[0]))
#         coords[1] = min(range(-97,-17,COORD_INT), key=lambda x:abs(x-coords[1]))
#         if date not in set(df['Date'].values):
#             avhrr = cdf4_to_pd(find_date(date),COORD_INT)
#             avhrr['Date'] = date
#             avhrr[['Max Wind','Min Pressure','Radius']] = -1
#             avhrr['Birth'] = 0
#             avhrr = avhrr[['Date','Latitude','Longitude','SST','SST Deviation','Max Wind','Min Pressure','Radius','Birth']]
#             df = pd.concat([df,avhrr], ignore_index=True)
#         df.loc[(df['Date'] == date) & (df['Latitude'] == coords[0]) & (df['Longitude'] == coords[1]), ['Max Wind','Min Pressure','Radius','Birth']] = birth_entry.max_wind,birth_entry.min_pressure,birth_entry.radius,1
#     df.to_csv('birth_input.csv',mode='a',index=False,header=False)



df = pd.read_csv('birth_input.csv')

# BINARY MODEL
def birth_binary_trans(df,sst):
    continuous_features = ['Latitude','Longitude','SST','SST Deviation']
    target = 'Birth'

    X_cont = df[continuous_features].values
    y = df[target].values

    scaler_x = StandardScaler()
    X_cont = scaler_x.fit_transform(X_cont)

    if os.path.exists("binary_birth_tf.pth") == False:
        X_cont = torch.tensor(X_cont, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float).unsqueeze(1)

        model = TabTransformer(      
            categories=[],
            num_continuous = len(continuous_features),                # number of continuous values
            dim = 32,                           # dimension, paper set at 32
            dim_out = 1,                        # binary prediction, but could be anything
            depth = 6,                          # depth, paper recommended 6
            heads = 8,                          # heads, paper recommends 8
            attn_dropout = 0.1,                 # post-attention dropout
            ff_dropout = 0.1,                   # feed forward dropout
            # mlp_hidden_mults = (4, 2),          # relative multiples of each hidden dimension of the last mlp to logits
            # mlp_act = nn.ReLU(),                # activation for final mlp, defaults to relu, but could be anything else (selu etc)
        )

        from torch.utils.data import Dataset, DataLoader
        import torch.optim as optim

        class HurricaneDataset(Dataset):
            def __init__(self, X_cont, y):
                self.X_cont = X_cont
                self.y = y

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return self.X_cont[idx], self.y[idx]

        X_cont_train, X_cont_val, y_train, y_val = train_test_split(
            X_cont, y, test_size=0.2, random_state=42
        )

        train_dataset = HurricaneDataset(X_cont_train, y_train)
        val_dataset = HurricaneDataset(X_cont_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        epochs = 7
        val_loss_dict = {}
        for epoch in range(epochs):
            model.train()
            for X_cont_batch, y_batch in train_loader:
                optimizer.zero_grad()
                dummy_cat = torch.empty((X_cont_batch.size(0), 0), dtype=torch.long)
                outputs = model(dummy_cat, X_cont_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_cont_batch, y_batch in val_loader:
                    dummy_cat = torch.empty((X_cont_batch.size(0), 0), dtype=torch.long)
                    outputs = model(dummy_cat, X_cont_batch)
                    val_loss += criterion(outputs, y_batch).item()
            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(val_loader):.4f}")
            val_loss_dict[epoch] = val_loss/len(val_loader)

        plt.figure(figsize=(10, 6))
        plt.plot(val_loss_dict.keys(), val_loss_dict.values(), label="Validation Loss", marker='o')
        plt.title("Binary Hurricane Birth Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
                   
        torch.save(model, "binary_birth_tf.pth")
        print("Model saved successfully at binary_birth_tf.pth!")

    if sst == None:
        return

    binary_model = torch.load("binary_birth_tf.pth")

    # Example predictions
    binary_model.eval()
    with torch.no_grad():
        sample_cont = torch.tensor(scaler_x.transform([sst]), dtype=torch.float)  # Example continuous feature
        dummy_cat = torch.empty((sample_cont.size(0), 0), dtype=torch.long)  # Empty categorical input
        prediction = binary_model(dummy_cat, sample_cont)
        return torch.sigmoid(prediction).item()

def daily_birth_prob(df,date):
    avhrr = cdf4_to_pd(find_date(date),COORD_INT)[['Latitude','Longitude','SST','SST Deviation']]
    avhrr['Birth Prob'] = avhrr.apply(
        lambda row: birth_binary_trans(df, [row['Latitude'],row['Longitude'],row['SST'],row['SST Deviation']]), axis=1
    )
    avhrr = avhrr.pivot(index='Latitude',columns='Longitude',values='Birth Prob')
    avhrr.index.name = "Latitude"
    avhrr.columns.name = "Longitude"
    return avhrr

# NEW HURRICANE CHARACTERISTIC MODEL
def new_hurricane_trans(df,sst):
    df = df[df['Birth'] == 1].drop(columns='Date')
    continuous_features = ['SST','SST Deviation']
    target = ['Max Wind','Min Pressure','Radius']

    X_cont = df[continuous_features].values
    y = df[target].values

    scaler_X = StandardScaler()
    X_cont = scaler_X.fit_transform(X_cont) # TEST DATA MUST BE SCALED X_test_scaled = scaler_X.transform(X_test) since both pre and post scaled

    scalers_y = [StandardScaler() for _ in range(y.shape[1])]
    y = np.hstack([scaler.fit_transform(y[:, i].reshape(-1, 1)) for i, scaler in enumerate(scalers_y)])

    if os.path.exists("continuous_birth_tf.pth") == False:
        X_cont = torch.tensor(X_cont, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)

        model = TabTransformer(      
            categories=[],                               # No categorical features
            num_continuous=len(continuous_features),     # Number of continuous features
            dim=32,                                      # Embedding dimension (paper suggests 32)
            dim_out=y.shape[1],                          # Output dimension equals the number of target columns
            depth=6,                                     # Depth of the transformer (paper recommends 6)
            heads=8,                                     # Number of attention heads (paper suggests 8)
            attn_dropout=0.1,                            # Attention dropout rate
            ff_dropout=0.1,                              # Feed-forward dropout rate
        )

        class HurricaneDataset(Dataset):
            def __init__(self, X_cont, y):
                self.X_cont = X_cont
                self.y = y

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return self.X_cont[idx], self.y[idx]

        X_cont_train, X_cont_val, y_train, y_val = train_test_split(
            X_cont, y, test_size=0.2, random_state=42
        )

        train_dataset = HurricaneDataset(X_cont_train, y_train)
        val_dataset = HurricaneDataset(X_cont_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        criterion = nn.MSELoss()  # Use Mean Squared Error Loss for regression
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        epochs = 20
        # val_loss_dict = {}
        for epoch in range(epochs):
            model.train()
            for X_cont_batch, y_batch in train_loader:
                optimizer.zero_grad()
                dummy_cat = torch.empty((X_cont_batch.size(0), 0), dtype=torch.long)
                outputs = model(dummy_cat, X_cont_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_cont_batch, y_batch in val_loader:
                    dummy_cat = torch.empty((X_cont_batch.size(0), 0), dtype=torch.long)
                    outputs = model(dummy_cat, X_cont_batch)
                    val_loss += criterion(outputs, y_batch).item()

            print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss/len(val_loader):.4f}")
            # val_loss_dict[epoch] = val_loss/len(val_loader)
        
        # plt.figure(figsize=(10, 6))
        # plt.plot(val_loss_dict.keys(), val_loss_dict.values(), label="Validation Loss", marker='o')
        # plt.title("Continuous Hurricane Birth Validation Loss")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        torch.save(model, "continuous_birth_tf.pth")
        print("Model saved successfully at continuous_birth_tf.pth!")

    if sst == None:
        return

    continuous_model = torch.load("continuous_birth_tf.pth")

    # Example predictions with the trained model
    continuous_model.eval()
    with torch.no_grad():
        sample_cont = torch.tensor(scaler_X.transform([sst]), dtype=torch.float)  # Example continuous feature values
        dummy_cat = torch.empty((sample_cont.size(0), 0), dtype=torch.long)  # Empty categorical input
        prediction = continuous_model(dummy_cat, sample_cont)
        y_pred_original = np.hstack([scaler.inverse_transform(prediction[:, i].reshape(-1, 1)) for i, scaler in enumerate(scalers_y)])
        return y_pred_original[0]

birth_binary_trans(df,None)

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



def hur_death_factor():
    wind = 0
    pres = 0
    radius = 0
    count = 0
    for hur in hurricanes_classed:
        death_entry = hur.entries[-1]
        if (death_entry.max_wind > 0) and (death_entry.min_pressure > 0) and (death_entry.radius > 0):
            wind += death_entry.max_wind
            pres += death_entry.min_pressure
            radius += death_entry.radius
            count += 1
    return (wind/count,pres/count,radius/count)





# day -> hurricane birth (binary) -> hurricane birth (characters) -> trajectory (death or landfall)


# Hurricane NICOLE
# nicole_day_prob = daily_birth_prob(df,'20221106')

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

# lons, lats = np.meshgrid(nicole_day_prob.columns, nicole_day_prob.index)
# values = nicole_day_prob.values

# plt.figure(figsize=(12, 8))
# ax = plt.axes(projection=ccrs.PlateCarree())  # Use PlateCarree projection
# ax.set_extent([-97,-17,4,47], crs=ccrs.PlateCarree())

# ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey')
# ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# # Plot heatmap
# heatmap = ax.pcolormesh(lons, lats, values, transform=ccrs.PlateCarree(), cmap="Reds",vmax=0.1)

# # Add color bar
# cbar = plt.colorbar(heatmap, orientation="vertical", pad=0.05, ax=ax)
# cbar.set_label("SST Values")

# # Add labels
# plt.title("Global SST Heatmap")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")

# plt.show()

# nicoles = []

# anicole = [[datetime(2022,11,6,12,0,0), 20.6,  -66.8,  30, 1005,  100],
#             [datetime(2022,11,6,18,0,0), 22.4,  -66.8,  35, 1005, 100],
#             [datetime(2022,11,7,0,0,0), 23.9,  -67.5,  35, 1005, 100],
#             [datetime(2022,11,7,6,0,0), 25.2,  -68.2,  40, 1004,  200],
#             [datetime(2022,11,7,12,0,0), 25.9,  -69.3,  40, 1002,   200],
#             [datetime(2022,11,7,18,0,0), 26.4,  -70.3,  40, 1001,  200],
#             [datetime(2022,11,8,0,0,0), 26.8,  -70.7,  40,  998,   200],
#             [datetime(2022,11,8,6,0,0), 27.5,  -71.2,  40,  996,  200],
#             [datetime(2022,11,8,12,0,0), 27.7,  -72.1,  45,  993,   80],
#             [datetime(2022,11,8,18,0,0), 27.6,  -73.3,  50,  991, 60],
#             [datetime(2022,11,9,0,0,0), 27.2,  -74.3,  60,  984,   60],
#             [datetime(2022,11,9,6,0,0), 26.8,  -75.3,  60,  986,   60],
#             [datetime(2022,11,9,12,0,0), 26.5,  -76.2,  55,  985,  20],
#             [datetime(2022,11,9,18,0,0), 26.5,  -77.3,  60,  985,   20],
#             [datetime(2022,11,10,0,0,0), 26.7,  -78.4,  65,  980,   20],
#             [datetime(2022,11,10,6,0,0), 27.3,  -79.8,  65,  980,   20],
#             [datetime(2022,11,10,12,0,0), 28.0,  -81.6,  55,  984,  110],
#             [datetime(2022,11,10,18,0,0), 29.0,  -82.8,  40,  989, 70],
#             [datetime(2022,11,11,0,0,0), 30.1,  -84.0,  35,  992,   70],
#             [datetime(2022,11,11,6,0,0), 31.2,  -84.6,  30,  996,   90],
#             [datetime(2022,11,11,12,0,0), 33.2,  -84.6,  25,  999,  250],
#             [datetime(2022,11,11,18,0,0), 35.4,  -83.8,  25, 1000,  400]]

# anicole = pd.DataFrame(anicole, columns=['Date','Lat','Lon','hurMaxWind','hurMinPressure','hurRadius'])


# # # 'Date','Lat','Lon','SST','SSTd','hurMaxWind','hurMinPressure','hurRadius' -> 'Next Lat','Next Lon','Next hurMaxWind','Next hurMinPressure','Next hurRadius'

# pnicole1 = trajectory_trans_pred(df,Entry(datetime(2022,11,6,12,0,0),'pAL172022',0,('20.6N','66.8W'),30,1007,100.41,301.7,0.5,0))
# pnicole2 = trajectory_trans_pred(df,Entry(datetime(2022,11,6,12,0,0),'pAL172022',0,('20.6N','66.8W'),28.94,1005.51,99.2,301.7,0.5,0))
# pnicole3 = trajectory_trans_pred(df,Entry(datetime(2022,11,6,12,0,0),'pAL172022',0,('20.6N','66.8W'),26.73,1003.89,98.6,301.7,0.5,0))
# pnicoles = [pnicole1,pnicole2,pnicole3]

# nicoles = [anicole,pnicole1,pnicole2,pnicole3]

# table = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[]}
# for hur in pnicoles:
#     for i in range(10):
#         lat = abs((anicole.iloc[i]['Lat']) - (hur.iloc[i]['Lat']))
#         lon = abs((anicole.iloc[i]['Lon']) - (hur.iloc[i]['Lon']))
#         hypo = (lat**2+lon**2)**(1/2)
#         table[i+1].append(hypo)

# keys = list(table.keys())
# values = list(table.values())

# # Transpose the values to separate each index
# lines = list(zip(*values))

# # Plot each line
# plt.figure(figsize=(10, 6))
# for i, line in enumerate(lines):
#     plt.plot(keys, line, label=f"Index {i}")

# plt.xticks(range(min(keys), max(keys) + 1, 1))

# # Add labels, legend, and title
# plt.xlabel("Steps")
# plt.ylabel("Absolute Error (degrees)")
# plt.title('Absolute Error of "Nicole" Predictions')
# plt.grid(True)
# plt.show()

# # Create a figure and axis with Cartopy
# plt.figure(figsize=(14, 8))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.set_extent([-97,-17,4,47], crs=ccrs.PlateCarree())

# # Add features to the map
# ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
# ax.add_feature(cfeature.BORDERS, linestyle=':')
# ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey')
# ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# norm = mcolors.Normalize(vmin=10, vmax=70)

# for i,hur in enumerate(nicoles):
#     lat = hur['Lat'].values
#     lon = hur['Lon'].values
#     wind = hur['hurMaxWind'].values

#     colors = ['Greys','Purples','Blues','Greens']
#     scatter = ax.scatter(lon, lat, c=wind, cmap=colors[i], norm=norm,
#                          s=100, edgecolors='k', alpha=0.7, marker='o',label=f'Hurricane {i+1}') 

#     ax.plot(lon, lat, color='black', linewidth=1, linestyle='-', alpha=0.9)

# # Add a color bar to indicate wind speed
# cbar = plt.colorbar(scatter, orientation="vertical", pad=0.05, ax=ax)
# cbar.set_label("Max Wind Speed (mph)")

# # Title and labels
# plt.title("Hurricane Trajectory Colored by Wind Speed")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")

# plt.show()