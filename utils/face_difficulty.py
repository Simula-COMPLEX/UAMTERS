#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/12/12 下午11:00
# @Author  : 
# @File    : face_difficulty.py
# @Software: PyCharm
import os
import pickle
from pathlib import Path

full_list = ['Parade', 'Handshaking', 'People_Marching', 'Meeting', 'Group', 'Interview', 'Traffic', 'Stock_Market',
             'Award_Ceremony', 'Ceremony', 'Concerts', 'Couple', 'Demonstration', 'Family_Group', 'Festival', 'Picnic',
             'Shoppers', 'Soldier_Firing', 'Soldier_Patrol', 'Soldier_Drilling', 'Spa', 'Sports_Fan',
             'Students_Schoolkids', 'Riot', 'Surgeons', 'Waiter_Waitress', 'Worker_Laborer', 'Running', 'Baseball',
             'Basketball', 'Football', 'Soccer', 'Tennis', 'Ice_Skating', 'Dancing', 'Gymnastics', 'Swimming',
             'Car_Racing', 'Row_Boat', 'Aerobics', 'Balloonist', 'Jockey', 'Matador_Bullfighter',
             'Parachutist_Paratrooper', 'Greeting', 'Car_Accident', 'Celebration_Or_Party', 'Dresses', 'Photographers',
             'Raid', 'Rescue', 'Sports_Coach_Trainer', 'Voter', 'Angler', 'Hockey', 'people--driving--car', 'Funeral',
             'Street_Battle', 'Cheering', 'Election_Campain', 'Press_Conference']

levels = {
    'hard': full_list[:20],       # First 20 elements: Parade, Handshaking, ..., Soldier_Drilling
    'medium': full_list[20:40],   # Next 20 elements: Spa, Sports_Fan, ..., Swimming
    'easy': full_list[40:]        # Remaining elements: Car_Racing, Row_Boat, ..., Press_Conference
}


hard = [
    "Traffic",
    "Festival",
    "Parade",
    "Demonstration",
    "Ceremony",
    "People_Marching",
    "Basketball",
    "Shoppers",
    "Matador_Bullfighter",
    "Car_Accident",
    "Election_Campain",
    "Concerts",
    "Award_Ceremony",
    "Picnic",
    "Riot",
    "Funeral",
    "Cheering",
    "Soldier_Firing",
    "Car_Racing",
    "Voter"
]

medium = [
    "Stock_Market",
    "Hockey",
    "Students_Schoolkids",
    "Ice_Skating",
    "Greeting",
    "Football",
    "Running",
    "people--driving--car",
    "Soldier_Drilling",
    "Photographers",
    "Sports_Fan",
    "Group",
    "Celebration_Or_Party",
    "Soccer",
    "Interview",
    "Raid",
    "Baseball",
    "Soldier_Patrol",
    "Angler",
    "Rescue"
]

easy = [
    "Gymnastics",
    "Handshaking",
    "Waiter_Waitress",
    "Press_Conference",
    "Worker_Laborer",
    "Parachutist_Paratrooper",
    "Sports_Coach_Trainer",
    "Meeting",
    "Aerobics",
    "Row_Boat",
    "Dancing",
    "Swimming",
    "Family_Group",
    "Balloonist",
    "Dresses",
    "Couple",
    "Jockey",
    "Tennis",
    "Spa",
    "Surgeons"
]


def get_elements(
        path='/home/complexse/workspace/RoboSapiens/Social_Perception/libfacedetection.train/data/widerface/origimg'
             '/orig/images',
        level=easy, level_label='easy'):
    events = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    event_dict = {}
    for event in events:
        event_dict.update({event.split('--', 1)[1]: event})

    images_level = []
    for event in level:
        event_imgs_path = Path(path) / event_dict[event]
        images = list(event_imgs_path.glob('*.jpg'))
        images = [img.name for img in images]
        images_level = images_level + images
    print(images_level)
    print(len(images_level))

    with open(f'{level_label}.pkl', 'wb') as f:
        pickle.dump(images_level, f)


for level in levels.keys():
    get_elements(level=levels[level], level_label=level)
# print(len(easy), len(medium), len(hard))
