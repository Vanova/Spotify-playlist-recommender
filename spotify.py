#### Well hello there! #####
###### Spotify playlist recommender system##########
######## Assisting the users in identifying the songs they would like when a new playlist is introduced
############ Made by: Rachit Mishra ################


import os
import spotipy
## Importing the spotipy Library ##
##os.chdir("C:/Punisher/1 UTD/Spring 18/spotify shizzle")

### IMPORTS ##
import pandas as pd
import numpy as np

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## importing packages for visualization ##

import seaborn as sns
import spotipy.util as util

# initialize spotify api

spotify = spotipy.Spotify()
from spotipy.oauth2 import SpotifyClientCredentials

clientid = "97b72e4133f3495c88c50016f4ac3347"
secret = "3bb3891a7a68420a817cdd9a5a2a5552"
username= "rachit.mishra94"

client_cred_manager = SpotifyClientCredentials(client_id=clientid,
                                               client_secret=secret)


model = spotipy.Spotify(client_credentials_manager=client_cred_manager)
scope = 'user-library-read playlist-read-private'
#token = "Bearer BQBHbEieVIBtDz5QcD3M6LxfWXcscBYd7ZzStDjP_3wKHkEFjY4QqFV-82EcU-wNvBo_Fnawc7JMx9KDLx1WdaWj2gojr5RGZguzsyEPg5RkkBPLDbnum850dZ4e_xRu41DOWsfZWNBq1tlfzDkRF1j3SgPAs-0zFut3WCoj&refresh_token=AQCPILw9Y6szk_KjqSVIHueizz-AU4iEPZ2SnkxLhTjtlPgI3l0eh8As9tEcmq3RJoqEMYceU71bA-jeUeYWnUOo5WZCNgq9IQUB9CqPR0OwFwJUInJFNVuncqKeU6yddAY"

token = util.prompt_for_user_token(username, scope, client_id=clientid, client_secret=secret, redirect_uri="https://rachitmishra25.com/callback/")
if token:
    model = spotipy.Spotify(auth=token)
else:
    print("unable to get token for:", username)

## defining playlists

#playlist1 https://open.spotify.com/user/rachit.mishra94/playlist/1VzO6phA696s1CRvzMCgbQ
hiphop_playlist = model.user_playlist("rachit.mishra94",
                                      "1VzO6phA696s1CRvzMCgbQ")

other_playlist = model.user_playlist("fh6g7k0xv9nim7jbewm42in21",
                                     "4UMMRDG0FyMV0IDAeoFJza")

suds_playlist = model.user_playlist("sudarsh8998",
                                               "3jmoXYoGBHueNBvAES0PAc?si=8DfX3NzARmatkbxYBDUBsQ")

mt_playlist = model.user_playlist("marriah.talha",
                                  "7Cfb498pWbxLEkRsIk9qZl")


## READING THE TRACKS FROM THE PLAYLISTS

hip_hop_tracks = hiphop_playlist["tracks"]
rap_songs = hip_hop_tracks["items"]

while hip_hop_tracks['next']:
    hip_hop_tracks = model.next(hip_hop_tracks)

    for each_item in hip_hop_tracks["items"]:
        rap_songs.append(each_item)

rap_song_ids = []
print(len(rap_songs))

for val in range(len(rap_songs)):
    rap_song_ids.append(rap_songs[val]['track']['id'])
hip_hop_tracks

#### For the other playlist

akshat_tracks = other_playlist["tracks"]
akshat_songs = akshat_tracks["items"]

while akshat_tracks['next']:
    akshat_tracks = model.next(akshat_tracks)

    for each_item in akshat_tracks["items"]:
        akshat_songs.append(each_item)

other_song_ids = []
print(len(akshat_songs))

for val in range(len(akshat_songs)-610):
    other_song_ids.append(akshat_songs[val]['track']['id'])
akshat_tracks


#####!#

rap_artist = []
other_artist = []

for i in range(0, len(hip_hop_tracks)):
    print(hip_hop_tracks)

##### FEATURES ####

features = []
saved = []
temp = 0
for item in range(0, len(rap_song_ids), 5):
    audio_features = model.audio_features(rap_song_ids[item:item+5])

    for track in audio_features:
        features.append(track)
        track = rap_songs[temp]

        temp+=1

        features[-1]['trackPopularity'] = track['track']['popularity']
        features[-1]['artistPopularity'] = model.artist(track['track']['artists'][0]['id'])['popularity']
        features[-1]['target'] = 1


temp2 = 0
for item in range(0, len(other_song_ids), 5):
    audio_features = model.audio_features(other_song_ids[item:item+5])

    for track in audio_features:
        features.append(track)
        track = akshat_songs[temp]

        temp2+=1

        features[-1]['trackPopularity'] = track['track']['popularity']
        features[-1]['artistPopularity'] = model.artist(track['track']['artists'][0]['id'])['popularity']
        features[-1]['target'] = 0


#### DEFINING THE TRAINING DATA ####

train = pd.DataFrame(features)

red_blue = ['#19B5FE', '#EF4836']
palette = sns.color_palette(red_blue)
sns.set_palette(palette)
sns.set_style('white')

# split the tr
train, test = train_test_split(train, test_size=0.2)
print("train size: {}, Test size: {}".format(len(train), len(test)))

## Breaking down the data into positive and negative
## features

pos_tempo = train[train['target'] == 1]['tempo']
neg_tempo = train[train['target'] == 0]['tempo']
pos_dance = train[train['target'] == 1]['danceability']
neg_dance = train[train['target'] == 0]['danceability']
pos_duration = train[train['target'] == 1]['duration_ms']
neg_duration = train[train['target'] == 0]['duration_ms']
pos_loudness = train[train['target'] == 1]['loudness']
neg_loudness = train[train['target'] == 0]['loudness']
pos_speechiness = train[train['target'] == 1]['speechiness']
neg_speechiness = train[train['target'] == 0]['speechiness']
pos_valence = train[train['target'] == 1]['valence']
neg_valence = train[train['target'] == 0]['valence']
pos_energy = train[train['target'] == 1]['energy']
neg_energy = train[train['target'] == 0]['energy']
pos_acousticness = train[train['target'] == 1]['acousticness']
neg_acousticness = train[train['target'] == 0]['acousticness']
pos_key = train[train['target'] == 1]['key']
neg_key = train[train['target'] == 0]['key']
pos_instrumentalness = train[train['target'] == 1]['instrumentalness']
neg_instrumentalness = train[train['target'] == 0]['instrumentalness']
pos_popularity = train[train['target'] == 1]['trackPopularity']
neg_popularity = train[train['target'] == 0]['trackPopularity']


print(pos_popularity)
print(neg_popularity)


### Tracks figures/plots/EDA visualizations

from matplotlib import pyplot as plt

figure = plt.figure(figsize=(12,8))

plt.title("Pos/neg distribution ")
pos_energy.hist(alpha=0.8, bins = 50, label='positive')
neg_energy.hist(alpha=0.8, bins = 50, label ='negative')

plt.legend(loc = 'upper-right')



fig2 = plt.figure(figsize=(15,15))

#Danceability
ax3 = fig2.add_subplot(331)
ax3.set_xlabel('Danceability metric')
ax3.set_ylabel('Count')
ax3.set_title('Danceability v/s Like')
pos_dance.hist(alpha= 0.5, bins=30)
ax4 = fig2.add_subplot(331)
neg_dance.hist(alpha= 0.5, bins=30)

#Duration_ms
ax5 = fig2.add_subplot(332)
ax5.set_xlabel('Duration metric')
ax5.set_ylabel('Count')
ax5.set_title('Duration v/s Like')
pos_duration.hist(alpha= 0.5, bins=30)
ax6 = fig2.add_subplot(332)
neg_duration.hist(alpha= 0.5, bins=30)

#Loudness
ax7 = fig2.add_subplot(333)
ax7.set_xlabel('Loudness metric')
ax7.set_ylabel('Count')
ax7.set_title('Loudness v/s Like')
pos_loudness.hist(alpha= 0.5, bins=30)
ax8 = fig2.add_subplot(333)
neg_loudness.hist(alpha= 0.5, bins=30)

#Speechiness
ax9 = fig2.add_subplot(334)
ax9.set_xlabel('Speechiness factor')
ax9.set_ylabel('Count')
ax9.set_title('Speechiness v/s Like')
pos_speechiness.hist(alpha= 0.5, bins=30)
ax10 = fig2.add_subplot(334)
neg_speechiness.hist(alpha= 0.5, bins=30)

#Valence
ax11 = fig2.add_subplot(335)
ax11.set_xlabel('Valence factor')
ax11.set_ylabel('Count')
ax11.set_title('Valence v/s Like')
pos_valence.hist(alpha= 0.5, bins=30)
ax12 = fig2.add_subplot(335)
neg_valence.hist(alpha= 0.5, bins=30)

#Energy
ax13 = fig2.add_subplot(336)
ax13.set_xlabel('Energy metric')
ax13.set_ylabel('Count')
ax13.set_title('Energy v/s Like')
pos_energy.hist(alpha= 0.5, bins=30)
ax14 = fig2.add_subplot(336)
neg_energy.hist(alpha= 0.5, bins=30)

#Key
ax15 = fig2.add_subplot(337)
ax15.set_xlabel('Key')
ax15.set_ylabel('Count')
ax15.set_title('Song Key Like Distribution')
pos_key.hist(alpha= 0.5, bins=30)
ax16 = fig2.add_subplot(337)
neg_key.hist(alpha= 0.5, bins=30)

#Key
ax15 = fig2.add_subplot(338)
ax15.set_xlabel('Popularity')
ax15.set_ylabel('Count')
ax15.set_title('Popularity Distribution')
pos_popularity.hist(alpha= 0.5, bins=30)
ax16 = fig2.add_subplot(338)
neg_popularity.hist(alpha= 0.5, bins=30)



features = ["danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness","duration_ms"]


x_train = train[features]
y_train = train["target"]

x_test = test[features]
y_test = test["target"]


c = DecisionTreeClassifier(min_samples_split=25)
dt = c.fit(x_train, y_train)
import io
import pydotplus
from scipy import misc


def show_tree(InputTree, features, path):
    f = io.StringIO()
    tree.export_graphviz(InputTree, out_file=f, feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img = misc.imread(path)
    plt.rcParams["figure.figsize"] = (20,20)
    plt.imshow(img)

show_tree(dt, features, "dec_tree2.png")

y_pred = c.predict(x_test)
score = accuracy_score(y_test, y_pred) * 100
print("Accuracy using Decision Tree: ", round(score, 1), "%")

## MLP classifier - 60% accuracy
# from sklearn.neural_network import MLPClassifier
# mlp = MLPClassifier()
# mlp.fit(x_train, y_train)
# mlp_pred = mlp.predict(x_test)
# score = accuracy_score(y_test, mlp_pred) * 100
# print("Accuracy using mlp Tree: ", round(score, 1), "%")

## QDA - 87.5% accuracy
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# qda = QuadraticDiscriminantAnalysis()
# qda.fit(x_train, y_train)
# qda_pred = qda.predict(x_test)
# score = accuracy_score(y_test, qda_pred)*100
# print("Accuracy using qda: ", round(score, 1), "%")

##Gradient Boosting
# from sklearn.ensemble import GradientBoostingClassifier
# gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, max_depth=1, random_state=0)
# gbc.fit(x_train, y_train)
# predicted = gbc.predict(x_test)
# score = accuracy_score(y_test, predicted)*100
# print("Accuracy using Gbc: ", round(score, 1), "%")


##
#############
#######################
###############################
### REQUEST TO modify playlist

username = "rachit.mishra94"
scope = 'playlist-modify-private playlist-modify-public playlist-read-private user-library-read'
# token = util.prompt_for_user_token(username, scope)
token = util.prompt_for_user_token(username, scope, client_id=clientid, client_secret=secret, redirect_uri="https://rachitmishra25.com/callback/")

if token:
    sp = spotipy.Spotify(auth=token)
new_playlist_check = sp.user_playlist("sudarsh8998", "3jmoXYoGBHueNBvAES0PAc?si=8DfX3NzARmatkbxYBDUBsQ")
#new_playlist_check = sp.user_playlist("fh6g7k0xv9nim7jbewm42in21","4UMMRDG0FyMV0IDAeoFJza")
#new_playlist_check = sp.user_playlist("marriah.talha", "7Cfb498pWbxLEkRsIk9qZl")

mt_tracks = new_playlist_check["tracks"]
mt_songs = mt_tracks["items"]

while mt_tracks['next']:
    mt_tracks = sp.next(mt_tracks)

    for each_song in mt_tracks["items"]:
        mt_songs.append(each_song)
## 224 songs in total in m.t space
mt_songs_ids = []
print(len(mt_songs))

for i in range(len(mt_songs)):
    mt_songs_ids.append(mt_songs[i]['track']['id'])

## features of new playlist
mt_new_features = []
j = 0

for  i in range(0, len(mt_songs_ids)):
    songs = sp.audio_features(mt_songs_ids[i])
    for song in songs:
        song['song_title'] = mt_songs[j]['track']['name']
        song['artist'] = mt_songs[j]['track']['artists'][0]['name']
        j+=1
        mt_new_features.append(song)

print(len(mt_new_features))

test_playlist = pd.DataFrame(mt_new_features)

#### tIME FOR PREDICTION #####


# using gradient boosting
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=.1, max_depth=1, random_state=0)
gbc.fit(x_train, y_train)
predicted = gbc.predict(x_test)
score = accuracy_score(y_test, predicted)*100
print("Accuracy using Gbc: ", round(score, 1), "%")
predictor = gbc.predict(test_playlist[features])


## using the original DT with 90% accuracy
dt.fit(x_train, y_train)
predicted = dt.predict(x_test)
score = accuracy_score(y_test, predicted)*100

#print("Accuracy using Gbc: ", round(score, 1), "%")

predictor = dt.predict(test_playlist[features])

songs_i_like  = 0
temp = 0

for val in predictor:
    if(val==1):
        print("The song is: "+ test_playlist["song_title"][temp] + ", Artist: " + test_playlist["artist"][temp])
        songs_i_like+=1
    temp+=1

#print("Can relate to: " + songs_i_like/temp + "of the playlist")


## Sampled playlists used for testing
###### 1. Akshat's playlist - 57/710 songs
###### 2. m.t space
#### Sud's playlist

