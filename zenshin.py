import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

class Zenshin():
    
    def __init__(self, cl_model='KMeans', reg_model='Lasso', reg_nn = 100, verbose=True):
        self.cl_model = None
        self.reg_model = None
        self.reg_nn = reg_nn
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_train_to_scale = None
        self.X_train_scaled = None
        
        if cl_model == 'KMeans':
            self.cl_model = KMeans(init="k-means++", n_init=2000, n_clusters=5, max_iter=50000)
        elif cl_model == 'AgglomerativeClustering':
            self.cl_model = AgglomerativeClustering(n_clusters=5)
        else:
            raise Exception("I possibili Modelli di Clustering sono: [KMeans, AgglomerativeClustering]")

        if reg_model == 'Lasso':
            self.reg_model = Lasso(tol=1e-5)
        elif reg_model == 'Ridge':
            self.reg_model = Ridge()
        else:
            raise Exception("I possibili Modelli di Regressione sono: [Lasso, Ridge]")
        
        print("Benvenuto in Zenshin.\n")
        print("*** Il Sistema utilizzerà il Modello di Regressione \'%s\', supportato dal Modello di Clustering \'%s\'." % (reg_model, cl_model))
        self.load_training_sets()

        self.train_cl_model()

    def load_training_sets(self):
        X_train = pd.read_csv("C:/Users/Riccardo De Cesaris/Desktop/Progetto ML x Football/Datasets/TOTAL/Train Set 1722.csv")
        X_train = X_train[["Player", "Pos", "Squad", "Age", "Season", "Goals", "xG", "Shots on Target", "Shots", "Att Pen", "Offsides",
                  "GCA", "Carries into Penalty Area", "PK Attempted", "PK Made", "Att 3rd", "GCA TO to Goal", "Take-Ons Attempted", "Take-Ons Successful",
                  "GCA Shot to Goal", "Goals Scored while on Pitch", "Carries into Final 1/3", "xGS while on Pitch", "Matches Played", "G/Shots on Target",
                   "G/Shot", "Minutes", "Shots on Target%", "Shots on Target/90", "Shots/90", "Mid 3rd", "Def 3rd", "Def Pen"]]
        X_train = X_train[(X_train["Matches Played"] >= 5) & (X_train["Minutes"] >= 343)]
        X_train = X_train.dropna(how="any")
        X_train.reset_index(drop=True, inplace = True)

        self.X_train = X_train
        self.X_train_to_scale = self.X_train[["xG", "Shots on Target", "Shots", "Att Pen", "Offsides",
                  "GCA", "Carries into Penalty Area", "PK Attempted", "PK Made", "Att 3rd", "GCA TO to Goal", "Take-Ons Attempted", "Take-Ons Successful",
                  "GCA Shot to Goal", "Goals Scored while on Pitch", "Carries into Final 1/3", "xGS while on Pitch", "Matches Played", "G/Shots on Target",
                   "G/Shot", "Minutes", "Shots on Target%", "Shots on Target/90", "Shots/90", "Mid 3rd", "Def 3rd", "Def Pen"]]
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_to_scale)

        if self.verbose:
            print("*** Il Training Set, caricato con successo, ha le seguenti dimensioni:", self.X_train_scaled.shape)

    def train_cl_model(self):
        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(self.X_train_scaled)

        start = datetime.now()
        self.cl_model.fit(X_train_pca)
        end = datetime.now()
        print("*** Tempo impiegato per l'Addestramento del Modello di Clustering:", (end-start))
    
    def build_regression_dataset(self, old_neighbors, scaled_vector):
        # Calcolo delle Distanze
        distances = np.empty(shape=(old_neighbors.shape[0]))
        for idx, i in zip(old_neighbors.index.values, range(0, old_neighbors.shape[0])):
            neighbor = self.X_train_scaled[idx]
            distances[i] = np.linalg.norm((scaled_vector-neighbor), ord=2)
        
        # Normalizzazione delle Distanze
        sum = np.sum(distances)
        for i in range(0, len(distances)):
            distances[i] = distances[i] / sum
        distances = np.round(distances*self.reg_nn, decimals=0).astype(int)
        reg_nn = np.sum(distances)
        
        # Costruzione dei Dataset
        X_reg_dataset = np.zeros(shape=(reg_nn, self.X_train_scaled.shape[1]))     # Contiene i dati (Feature scalate) relativi ai giocatori identificati come "simili" al Giocatore di input
        y_reg_dataset = np.zeros(shape=(reg_nn, 1))                                # Contiene i gol segnati dai giocatori identificati come "simili" al Giocatore di input
        ds_counter = 0
        for idx, i in zip(old_neighbors.index.values, range(0, len(distances))):
            # Ogni "vecchio datapoint" del Giocatore di input è considerabile come un suo "simile".
            X_reg_dataset[ds_counter] = self.X_train_scaled[idx]
            y_reg_dataset[ds_counter] = old_neighbors.loc[idx]["Goals"]
            ds_counter += 1

            # Recuperiamo tutto il Cluster a cui appartiene il "vecchio datapoint" che stiamo considerando.
            label = self.cl_model.labels_[idx]
            indexes = np.where(self.cl_model.labels_ == label)[0].tolist()
            cluster = self.X_train_scaled[indexes]

            # A seconda della distanza "giocatore di input-vecchio datapoint" recuperiamo i NearestNeighbors del "vecchio datapoint".
            neighbors_finder = NearestNeighbors(n_neighbors=distances[i], metric='minkowski', p=2)
            neighbors_finder.fit(cluster)

            neighbors_found = neighbors_finder.kneighbors(X_reg_dataset[ds_counter-1].reshape(1,-1), return_distance=False)[0]
            neighbors_found = neighbors_found[1:] # Si rimuove il primo NearestNeighbor, che corrisponde al "vecchio datapoint" stesso (già inserito nel Dataset per la Regressione).

            for i in neighbors_found:
                dataset_index = indexes[i]
                X_reg_dataset[ds_counter] = self.X_train_scaled[dataset_index]
                y_reg_dataset[ds_counter] = self.X_train.loc[dataset_index]["Goals"]
                ds_counter += 1
        
        return X_reg_dataset, y_reg_dataset

    
    def predict(self, player_name):
        # Recupero dei "Vecchi Datapoint"
        old_neighbors = self.X_train[self.X_train["Player"] == player_name]
        if old_neighbors.shape[0] == 0:
            raise Exception("Non è possibile effettuare predizioni su un Giocatore di cui non si conosce la Carriera pregressa!")
        if self.verbose:
            print("Ecco i \"vecchi datapoint\" del Giocatore di input:")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
                display(old_neighbors)
        
        vector = np.zeros((1, self.X_train_scaled.shape[1]))

        ### TO DO: Struttura Grafica per l'inserimento dei Dati del Giocatore di Input ###

        scaled_vector = self.scaler.transform(vector)

        X_reg_dataset, y_reg_dataset = self.build_regression_dataset(old_neighbors, scaled_vector)
        
        hp = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
        grid = GridSearchCV(self.reg_model, hp, scoring='neg_mean_squared_error', cv=5)
        grid.fit(X=X_reg_dataset, y=y_reg_dataset)
        if isinstance(self.reg_model, Lasso):
            self.reg_model = Lasso(alpha=grid.best_params_['alpha'], tol=1e-5)
        else:
            self.reg_model = Ridge(alpha=grid.best_params_['alpha'])
        
        start = datetime.now()
        self.reg_model.fit(X=X_reg_dataset, y=y_reg_dataset)
        end = datetime.now()
        print("*** Tempo impiegato per l'Addestramento del Modello di Regressione:", (end-start))

        ### TO DO: Visualizzazione della Retta di Regressione, del Training Set e del Giocatore di Input (matplotlib) ###
        
        goals_predicted = None
        if isinstance(self.reg_model, Lasso):
            goals_predicted = self.reg_model.predict(scaled_vector)[0]
        else:
            goals_predicted = self.reg_model.predict(scaled_vector)[0][0]
        
        if goals_predicted < 0:
            print("*** Con i dati specificati, il Giocatore \"%s\" dovrebbe segnare 0 Goal nella Prossima Stagione.\n" % player_name)
            print("*** Attenzione! Il Modello ha in realtà predetto che il Giocatore segni %f Goal, ma ciò è (ovviamente) impossibile." % goals_predicted)
            print("*** Considera questo risultato come indicatore di quanto lo stile di gioco specificato sia lontano dall'attitudine al Goal.")
        else:
            print("*** Con i dati specificati, il Giocatore \"%s\" dovrebbe segnare %f Goal nella Prossima Stagione." % (player_name, goals_predicted))