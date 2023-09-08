import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV

class Zenshin():
    
    def __init__(self, mode=1, verbose=True):
        self.cl_model = None
        self.reg_model = None
        self.desc = []
        self.reg_nn = None
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_train_to_scale = None
        self.X_train_scaled = None

        if mode == 1: #KMeans + Lasso
            self.cl_model = KMeans(init="k-means++", n_init=2000, n_clusters=5, max_iter=50000)
            self.reg_model = Lasso(tol=1e-5)
        if mode == 2: #Spectral + Ridge
            self.cl_model = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', n_neighbors=100, assign_labels='kmeans', n_init=50)
            self.reg_model = Ridge(solver="cholesky")
        if mode == 3: #Spectral + Lasso
            self.cl_model = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', n_neighbors=100, assign_labels='kmeans', n_init=50)
            self.reg_model = Lasso(tol=1e-5)
        if mode == 4: #Agglomerative + Lasso
            self.cl_model = AgglomerativeClustering(n_clusters=5)
            self.reg_model = Lasso(tol=1e-5)
        if mode == 5: #KMeans + LinearSVR
            self.cl_model = KMeans(init="k-means++", n_init=2000, n_clusters=5, max_iter=50000)
            self.reg_model = LinearSVR()
        if mode == 6: #Spectral + LinearSVR
            self.cl_model = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', n_neighbors=100, assign_labels='kmeans', n_init=50)
            self.reg_model = LinearSVR()
        if mode == 7: #Agglomerative + LinearSVR
            self.cl_model = AgglomerativeClustering(n_clusters=5)
            self.reg_model = LinearSVR()
        if mode >= 8 or mode <= 0:
            raise Exception("I possibili modelli sono: " + str(list(range(0,8))))
        
        if isinstance(self.cl_model, KMeans):                     self.desc.append('KMeans')
        if isinstance(self.cl_model, AgglomerativeClustering):    self.desc.append('Agglomerative Clustering')
        if isinstance(self.cl_model, SpectralClustering):         self.desc.append('Spectral Clustering')
        
        if isinstance(self.reg_model, Ridge):                     self.desc.append('Ridge')
        if isinstance(self.reg_model, Lasso):                     self.desc.append('Lasso')
        if isinstance(self.reg_model, LinearSVR):                 self.desc.append('LinearSVR')
        
        print("Benvenuto in Zenshin.\n")
        print("*** Il Sistema utilizzerà il Modello di Regressione \'%s\', supportato dal Modello di Clustering \'%s\'." % (self.desc[1], self.desc[0]))
        
        self.load_training_sets()
        self.train_cl_model()

    def load_training_sets(self):
        X_train = pd.read_csv("dataset/Training Set 1723.csv")
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
        self.reg_nn = round(3*np.sqrt(self.X_train_scaled.shape[0]))

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
        distances = 1/distances
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

    def create_player_table(self, player_name):
        features_dict = {
                        "xG": [0], 
                        "Shots on Target": [0], 
                        "Shots": [0], 
                        "Att Pen": [0], 
                        "Offsides": [0],
                        "GCA": [0], 
                        "Carries into Penalty Area": [0], 
                        "PK Attempted": [0], 
                        "PK Made": [0], 
                        "Att 3rd": [0], 
                        "GCA TO to Goal": [0], 
                        "Take-Ons Attempted": [0], 
                        "Take-Ons Successful": [0],
                        "GCA Shot to Goal": [0], 
                        "Goals Scored while on Pitch": [0], 
                        "Carries into Final 1/3": [0], 
                        "xGS while on Pitch": [0], 
                        "Matches Played": [0], 
                        "G/Shots on Target": [0],
                        "G/Shot": [0],
                        "Minutes": [0],
                        "Shots on Target%": [0],
                        "Shots on Target/90": [0],
                        "Shots/90": [0], 
                        "Mid 3rd": [0], 
                        "Def 3rd": [0], 
                        "Def Pen": [0]
                        }
        filepath = "predictions/" + f'{player_name}' + " " + str(self.desc) + " Prediction.txt"
        json.dump(features_dict, open(filepath, "w"))

        return filepath
    
    def train_reg_model(self, X_reg_dataset, y_reg_dataset):
        if isinstance(self.reg_model, Lasso) or isinstance(self.reg_model, Ridge):
            hp = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
            grid = GridSearchCV(self.reg_model, hp, scoring='neg_mean_squared_error', cv=5)
            grid.fit(X=X_reg_dataset, y=y_reg_dataset)
            
            if isinstance(self.reg_model, Lasso):
                self.reg_model = Lasso(alpha=grid.best_params_['alpha'], tol=1e-5)
            if isinstance(self.reg_model, Ridge):
                self.reg_model = Ridge(alpha=grid.best_params_['alpha'], solver='cholesky')
        
        if isinstance(self.reg_model, LinearSVR):
            hp = {
            'epsilon': [0.0, 0.1, 0.2, 0.3, 0.5],
            'C': [0.1, 1, 10, 50, 100]
            }
            grid = GridSearchCV(self.reg_model, hp, scoring='neg_mean_squared_error', cv=5)
            grid.fit(X=X_reg_dataset, y=y_reg_dataset)

            self.reg_model = LinearSVR(C=grid.best_params_['C'], epsilon=grid.best_params_['epsilon'])
        
        start = datetime.now()
        self.reg_model.fit(X=X_reg_dataset, y=y_reg_dataset)
        end = datetime.now()
        print("*** Tempo impiegato per l'Addestramento del Modello di Regressione:", (end-start))

    def predict(self, player_name):
        # Recupero dei "Vecchi Datapoint"
        old_neighbors = self.X_train[self.X_train["Player"] == player_name]
        if old_neighbors.shape[0] == 0:
            raise Exception("Non è possibile effettuare predizioni su un Giocatore di cui non si conosce la Carriera pregressa!")
        if self.verbose:
            print("Ecco i \"vecchi datapoint\" del Giocatore di input:", end = '')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3):
                display(old_neighbors)
        
        stats_filepath = self.create_player_table(player_name)

        go_to_prediction = False
        print("*** Inserisci nel file \'" + stats_filepath + "\' le statistiche per cui desideri la predizione: effettuata questa operazione, digita \"DONE\"")
        while(go_to_prediction is False):
            string = input()
            if string == "DONE":
                go_to_prediction = True
        
        features_dict = json.load(open(stats_filepath))
        vector = pd.DataFrame.from_dict(features_dict)
        scaled_vector = self.scaler.transform(vector)

        X_reg_dataset, y_reg_dataset = self.build_regression_dataset(old_neighbors, scaled_vector)

        self.train_reg_model(X_reg_dataset, y_reg_dataset)
        
        goals_predicted = None
        if isinstance(self.reg_model, Lasso) or isinstance(self.reg_model, LinearSVR):
            goals_predicted = self.reg_model.predict(scaled_vector)[0]
        else:
            goals_predicted = self.reg_model.predict(scaled_vector)[0][0]
        
        if goals_predicted < 0:
            print("*** Con i dati specificati, il Giocatore \"%s\" dovrebbe segnare 0 Goal nella Prossima Stagione.\n" % player_name)
            print("*** Attenzione! Il Modello ha in realtà predetto che il Giocatore segni %f Goal, ma ciò è (ovviamente) impossibile." % goals_predicted)
            print("*** Considera questo risultato come indicatore di quanto le statistiche specificate siano lontane dall'attitudine al Goal.")
        else:
            print("*** Con i dati specificati, il Giocatore \"%s\" dovrebbe segnare %f Goal nella Prossima Stagione." % (player_name, goals_predicted))
        
        with open(stats_filepath, "a") as f:
            f.write("\n\nGoal Predetti: " + str(goals_predicted))