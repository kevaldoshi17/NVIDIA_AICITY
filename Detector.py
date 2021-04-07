#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter, defaultdict
from kneed import KneeLocator
from sklearn.cluster import KMeans
from matplotlib.patches import Rectangle
from vid_utils import *

import numpy as np 
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from os import path


# In[2]:


with open('result.json', 'r') as f:
    D = json.load(f)


#  ## Extract Objects

# In[3]:


All_Cords = extract_objects(D)


# ## Extract Cases

# In[4]:


AT = extract_cases(All_Cords)


# ## Detect Change in Camera

# In[6]:


Base = "processed_images/"
if path.exists("change.npy"):
    change_cam,loc,Cstat = np.load("change.npy",allow_pickle=True)
else:
    change_cam, loc,Cstat = change_detect(Base)
    np.save("change.npy",[change_cam,loc,Cstat])


# In[7]:


PT = list(set(AT) - set(change_cam))


# ## Case 1: Extract ROI

# In[8]:


if path.exists("centers1.npy"):
    Centers = np.load("centers1.npy",allow_pickle=True)
else:
    Centers = extract_roi(PT,All_Cords)
    np.save("centers1.npy",Centers)


# In[9]:


len(Centers)


# ## Case 1: Extract Bounds

# In[10]:




if path.exists("bounds1.npy"):
    Bounds = np.load("bounds1.npy",allow_pickle=True)
else:
    Bounds = extract_bounds(Centers,PT,All_Cords)
    np.save("bounds1.npy",Bounds)


# In[11]:


len(Bounds)


# ## Case 1: Backtracking

# In[12]:


Base = "ori_images/"

if path.exists("result1.npy"):
    Times, Stat = np.load("result1.npy",allow_pickle=True)
else:
    Times, Stat = backtrack(Bounds,PT,Base)
    np.save("result1.npy",[Times,Stat])


# ## Case 2: Extract ROI 

# In[13]:


if path.exists("centers2.npy"):
    Centers2 = np.load("centers2.npy",allow_pickle=True)
else:
    Centers2 = extract_roi1(change_cam,All_Cords,loc)
    np.save("centers2.npy",Centers2)


# ## Case 2: Extract Bounds

# In[14]:


if path.exists("bounds2.npy"):
    Bounds2 = np.load("bounds2.npy",allow_pickle=True)
else:
    Bounds2 = extract_bounds1(Centers2,change_cam,loc,All_Cords)
    np.save("bounds2.npy",Bounds2)


# ## Case 2: Backtracking

# In[15]:


len(Centers),len(Centers2)


# In[16]:


len(Bounds),len(Bounds2)


# In[17]:


Base = "ori_images/"


if path.exists("result2.npy"):
    Times2, Stat2 = np.load("result2.npy",allow_pickle=True)
else:
    Times2, Stat2 = backtrack1(Bounds2,Base)
    np.save("result2.npy",[Times2,Stat2])


# In[18]:


Times = {key:val for key, val in Times.items() if val != 999}
Times = {key:val for key, val in Times.items() if val >= 40}

Times2 = {key:val for key, val in Times2.items() if val != 999}
Times2 = {key:val for key, val in Times2.items() if val >= 40}


# In[23]:


file1 = open("Result" + ".txt","w")
for x in Times:
    file1.write('{0:2d} {1:3d} {2:1d}'.format(x,int(Times[x]),1))
    file1.write("\n")
    
for x in Times2:
    file1.write('{0:2d} {1:3d} {2:1d}'.format(x,int(Times2[x]),1))
    file1.write("\n")
    
file1.close()


# In[ ]:




