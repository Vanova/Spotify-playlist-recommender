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


other_tracks = other_playlist["tracks"]
other_songs = other_tracks["items"]

while other_tracks['next']:
    other_tracks = model.next(other_tracks)

    for each_item in other_tracks["items"]:
        other_songs.append(each_item)

other_song_ids = []
print(len(other_songs))

for val in range(len(other_songs)-610):
    other_song_ids.append(other_songs[val]['track']['id'])
other_tracks


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
        track = other_songs[temp]

        temp2+=1

        features[-1]['trackPopularity'] = track['track']['popularity']
        features[-1]['artistPopularity'] = model.artist(track['track']['artists'][0]['id'])['popularity']
        features[-1]['target'] = 0


#### DEFINING THE TRAINING DATA ####

train = pd.DataFrame(features)


# split the tr
train, test = train_test_split(train, test_size=0.2)
print("train size: {}, Test size: {}".format(len(train), len(test)))






