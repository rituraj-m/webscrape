# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:51:33 2019

@author: Rituraj
"""

import pandas as pd
import requests
import numpy as np
from bs4 import BeautifulSoup
import pickle

res = requests.get("http://www.estesparkweather.net/archive_reports.php?date=200901")
soup = BeautifulSoup(res.content,'lxml')
table = soup.find_all('table')

df = pd.read_html(str(table))
arr = np.array(df)
i=0
for i in range(len(arr)): 
    mat = np.asmatrix(arr[i])
    #print(mat)
    Tmat = mat.T
    #print(Tmat)
    Tarr = np.array(Tmat)
    #print(Tarr.item(2))
    j=0
    for j in range(Tarr.size):
        items = Tarr.item(j)
        my_list = []
        my_list.append(items)
        df = pd.DataFrame(my_list) 
        Idict = df.to_dict()
        pd.to_pickle(Idict,'my_file.pk')
        infile = open('my_file.pk','rb')
        new_dict = pickle.load(infile, encoding='bytes')
        
    
    