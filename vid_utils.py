from collections import Counter
from kneed import KneeLocator
import numpy as np
from sklearn.cluster import KMeans
from natsort import natsorted
import cv2
import pandas as pd
import os
from skimage import measure
from scipy.signal import savgol_filter



def Kmeans_clu(data,k):

    kmeans = KMeans(n_clusters=k, max_iter=10000, n_init=10).fit(data) ##Apply k-means clustering
    labels = kmeans.labels_
    clu_centres = kmeans.cluster_centers_
    z={i: np.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}
    
    return clu_centres, labels ,z,kmeans
    
def reject_outliers(data):
    m = 2
    u = np.mean(data)
    s = np.std(data)
    filtered = [e for e in data if (u - 2 * s < e < u + 2 * s)]
    return np.array(filtered)


def check_continual(stat,k):
    Found = False
    inx = (stat>0.5).astype(int)
    for ix in range(0,len(inx)-k,k):
        if np.min((inx[ix:ix+k])) == 1:
            Found = True
    return Found
    
    
def nms(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes	
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0] - (boxes[:,3]/2)
    y1 = boxes[:,1] - (boxes[:,2]/2)
    x2 = boxes[:,0] + (boxes[:,3]/2)
    y2 = boxes[:,1] + (boxes[:,2]/2)

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
#         print(overlap)

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
#     print(pick)
    return boxes[pick]


def extract_objects(D):
    
    c = 62
    valid_list = ['bus','truck','person','car','boat']
    All_Cords = list()
    for vid in range(1,63):

        Vid_Cords = list()
        for file in range(22,266):
            Frame_Cords = list()

            if (len(D[c]['objects']))!=0:
                for bound in D[c]['objects']:
                    if bound['name'] in valid_list:
                        x = bound['relative_coordinates']['center_x']*400
                        y = bound['relative_coordinates']['center_y']*410
                        h = bound['relative_coordinates']['height']*410
                        w = bound['relative_coordinates']['width']*400
                        f = file 
                        if w*h > 16 and w>4 and h>4:
                            Frame_Cords.append([x,y,h,w,f])

            c+=1

            if (len(D[c]['objects']))!=0:
                for bound in D[c]['objects']:
                    if bound['name'] in valid_list:
                        x = bound['relative_coordinates']['center_x']*400 + 200
                        y = bound['relative_coordinates']['center_y']*410
                        h = bound['relative_coordinates']['height']*410
                        w = bound['relative_coordinates']['width']*400
                        f = file
                        if w*h > 16 and w> 4 and h>4:
                            Frame_Cords.append([x,y,h,w,f])

            c+=1

            if (len(D[c]['objects']))!=0:
                for bound in D[c]['objects']:
                    if bound['name'] in valid_list:
                        x = bound['relative_coordinates']['center_x']*400 + 400
                        y = bound['relative_coordinates']['center_y']*410
                        h = bound['relative_coordinates']['height']*410
                        w = bound['relative_coordinates']['width']*400
                        f = file
                        if w*h > 16 and w>4 and h>4:
                            Frame_Cords.append([x,y,h,w,f])

            c+=1

            Frame_Cords = nms(np.array(Frame_Cords),0.9)
            Vid_Cords.append(Frame_Cords)
        c+=67
        All_Cords.append(Vid_Cords)


    for vid in range(63,64):
        Vid_Cords = list()
        for file in range(22,232):
            Frame_Cords = list()

            if (len(D[c]['objects']))!=0:
                for bound in D[c]['objects']:
                    if bound['name'] in valid_list:
                        x = bound['relative_coordinates']['center_x']*400
                        y = bound['relative_coordinates']['center_y']*410
                        h = bound['relative_coordinates']['height']*410
                        w = bound['relative_coordinates']['width']*400
                        f = file
                        if w*h > 16 and w> 4 and h>4:
                            Frame_Cords.append([x,y,h,w,f])

            c+=1

            if (len(D[c]['objects']))!=0:
                for bound in D[c]['objects']:
                    if bound['name'] in valid_list:
                        x = bound['relative_coordinates']['center_x']*400 + 200
                        y = bound['relative_coordinates']['center_y']*410
                        h = bound['relative_coordinates']['height']*410
                        w = bound['relative_coordinates']['width']*400
                        f = file
                        if w*h > 16 and w>4 and h>4:
                            Frame_Cords.append([x,y,h,w,f])

            c+=1

            if (len(D[c]['objects']))!=0:
                for bound in D[c]['objects']:
                    if bound['name'] in valid_list:
                        x = bound['relative_coordinates']['center_x']*400 + 400
                        y = bound['relative_coordinates']['center_y']*410
                        h = bound['relative_coordinates']['height']*410
                        w = bound['relative_coordinates']['width']*400
                        f = file
                        if w*h > 16 and w>4 and h>4:
                            Frame_Cords.append([x,y,h,w,f])

            c+=1

            Frame_Cords = nms(np.array(Frame_Cords),0.9)
            Vid_Cords.append(Frame_Cords)

        c+=67
        All_Cords.append(Vid_Cords)
        

    for vid in range(64,101):
    
        Vid_Cords = list()
        for file in range(22,266):
            Frame_Cords = list()

            if (len(D[c]['objects']))!=0:
                for bound in D[c]['objects']:
                    if bound['name'] in valid_list:
                        x = bound['relative_coordinates']['center_x']*400
                        y = bound['relative_coordinates']['center_y']*410
                        h = bound['relative_coordinates']['height']*410
                        w = bound['relative_coordinates']['width']*400
                        f = file
                        if w*h > 16 and w>4 and h>4:
                            Frame_Cords.append([x,y,h,w,f])

            c+=1

            if (len(D[c]['objects']))!=0:
                for bound in D[c]['objects']:
                    if bound['name'] in valid_list:
                        x = bound['relative_coordinates']['center_x']*400 + 200
                        y = bound['relative_coordinates']['center_y']*410
                        h = bound['relative_coordinates']['height']*410
                        w = bound['relative_coordinates']['width']*400
                        f = file
                        if w*h > 16 and w>4 and h>4:
                            Frame_Cords.append([x,y,h,w,f])

            c+=1

            if (len(D[c]['objects']))!=0:
                for bound in D[c]['objects']:
                    if bound['name'] in valid_list:
                        x = bound['relative_coordinates']['center_x']*400 + 400
                        y = bound['relative_coordinates']['center_y']*410
                        h = bound['relative_coordinates']['height']*410
                        w = bound['relative_coordinates']['width']*400
                        f = file
                        if w*h > 16 and w>4 and h>4:
                            Frame_Cords.append([x,y,h,w,f])

            c+=1

            Frame_Cords = nms(np.array(Frame_Cords),0.9)
            Vid_Cords.append(Frame_Cords)

        c+=67
        All_Cords.append(Vid_Cords)
        
    return(All_Cords)



def extract_cases(All_Cords):
    maxes = list()
    for i in range(0,100):

        T = np.array(All_Cords[i], dtype=object)
        AT = [item for sublist in T for item in sublist]
        AT = np.array(np.array(AT)).reshape(-1,5)

        new_idx = list()
        diss = list()
        if AT.shape[0] > 10:
            y = []
            K = range(1,10)
            for k in K:
                km = KMeans(n_clusters=k,max_iter=10000, n_init=10)
                km = km.fit(AT[:,0:4])
                y.append(km.inertia_)

            x = range(1, len(y)+1)


            kn = KneeLocator(x, y, curve='convex', direction='decreasing')
            k = (kn.knee) + 1

            if k == None:
                k = 10

            center,lb,z,kmeans = Kmeans_clu(AT[:,0:4],k)
            max_count = Counter(lb)

            if AT.shape[0] > 1:
                mx = max_count[max(max_count, key=max_count.get)]
            else:
                mx = 0
        else:
            mx = 0
        maxes.append(mx)
        
    PT = list(np.where(np.array(maxes) > 20)[0] + 1)
    return(PT)

def extract_roi(PT,All_Cords):
    Centers = list()
    for i in PT: 

        Mask = np.load("Masks/Mas/" + str(i)+ ".npy")
        T = np.array(All_Cords[i-1])
        AT = [item for sublist in T for item in sublist]
        AT = np.array(np.array(AT)).reshape(-1,5)
        if AT.shape[0] > 20:
            y = []
            K = range(1,10)
            for k in K:
                km = KMeans(n_clusters=k, max_iter=10000,n_init=10)
                km = km.fit(AT[:,0:4])
                y.append(km.inertia_)

            x = range(1, len(y)+1)


            kn = KneeLocator(x, y, curve='convex', direction='decreasing')
            k = (kn.knee) + 1

            if k == None:
                k = 10

            center,lb,z,kmeans = Kmeans_clu(AT[:,0:4],k)
            vid_cent = list()
            for cent in center:
                y_m = np.max((0,int(cent[1])-10))
                y_ma = np.min((410,int(cent[1])+10))
                x_m = np.max((0,int(cent[0])-10))
                x_ma = np.min((800,int(cent[0])+10))
                if np.max(Mask[y_m:y_ma,x_m:x_ma]) == 1:               
                    vid_cent.append(cent)
            Centers.append(vid_cent)
            
    return(Centers)

def extract_roi1(cam_change,All_Cords,loc):
    Centersf = list()
    for i in range(0,len(cam_change)):
        Centers = list()
        T = np.array(All_Cords[cam_change[i]-1])
        AT = [item for sublist in T for item in sublist]
        AT = np.array(np.array(AT)).reshape(-1,5)
        imx = np.where(AT[:,4] >= 10*loc[i])[0][0]
        AT = AT[0:imx,:]

        Mask = np.load("Masks/Mas/" + str(cam_change[i])+ ".npy")

        if AT.shape[0] > 10:
            y = []
            K = range(1,10)
            for k in K:
                km = KMeans(n_clusters=k, max_iter=10000,n_init=10)
                km = km.fit(AT[:,0:4])
                y.append(km.inertia_)

            x = range(1, len(y)+1)
            kn = KneeLocator(x, y, curve='convex', direction='decreasing')
            k = (kn.knee) + 1

            if k == None:
                k = 10

            center,lb,z,kmeans = Kmeans_clu(AT[:,0:4],k)
            vid_cent = list()
            for cent in center:
                y_m = np.max((0,int(cent[1])-10))
                y_ma = np.min((410,int(cent[1])+10))
                x_m = np.max((0,int(cent[0])-10))
                x_ma = np.min((800,int(cent[0])+10))
                if np.max(Mask[y_m:y_ma,x_m:x_ma]) == 1:               
                    vid_cent.append(cent)

            Centers.append(vid_cent)
        Centersf.append(Centers)
    return Centersf

def extract_bounds(Centers,PT,All_Cords):
    
    count = 0
    Bounds = list()
    for centro in Centers:

        T = np.array(All_Cords[-1+PT[count]])
        AT = [item for sublist in T for item in sublist]
        AT = np.array(np.array(AT)).reshape(-1,5)

        count2 = 1
        for c in centro:
            Found = False
            idx = 0
            while not Found:
                if idx < len(AT):
                    if np.abs(AT[idx,0] - c[0]) < 25 and np.abs(AT[idx,1] - c[1]) < 25 and AT[idx,2] < 100 and int(AT[idx,2]) >7 and int(AT[idx,3]) >7:

                        Bounds.append([AT[idx,0:],PT[count],count2])
                        Found = True
                else:
                    Found = True
                idx+=1
            count2+=1
        count+=1
        
    return(Bounds)


def extract_bounds1(Centers,cam_change,loc,All_Cords):
    
    Bounds = list()
    for i in range(0,len(cam_change)):
        for centro in Centers[i]:

            T = np.array(All_Cords[cam_change[i]-1])
            AT = [item for sublist in T for item in sublist]
            AT = np.array(np.array(AT)).reshape(-1,5)
            imx = np.where(AT[:,4] >= 10*loc[i])[0][0]
            AT = AT[0:imx,:]

            count2 = 1
            for c in centro:
                Found = False
                idx = 0
                while not Found:
                    if idx < len(AT):
                        if np.abs(AT[idx,0] - c[0]) < 25 and np.abs(AT[idx,1] - c[1]) < 25 and AT[idx,2] < 100 and int(AT[idx,2]) >6 and int(AT[idx,3]) >6:
                            Bounds.append([AT[idx,0:],cam_change[i],count2])
                            Found = True
                    else:
                        Found = True
                    idx+=1
                count2+=1
    
    return (Bounds)
    


        
def change_detect(Base):
    
    All_stat3= list()
    Folders = natsorted(os.listdir(Base))
    for i in range(len(Folders)):

        base2 = Base + str(Folders[i]) + "/" 
        files = natsorted(os.listdir(base2))
        stat = list()

        for idx in range(1,len(files)-10,10):

            img0 = cv2.imread(base2 + files[idx])
            img1 = cv2.imread(base2 + files[idx+10])
            sad = 0
            if np.sum(img1) != 0 and np.sum(img0) !=0:
                sad = measure.compare_ssim(img0,img1,multichannel=True,win_size=3)
            else:
                sad = 0.99
            stat.append(np.max((0,sad)))
        All_stat3.append(stat)    
        
    cam_change = list()
    loc = list()
    for ss in range(len(All_stat3)):
#             if len(np.where(np.array(All_stat3[ss]) < 0.87)[0]) > 0:
        if np.mean(All_stat3[ss]) - np.min((All_stat3[ss])) > 0.06:
            cam_change.append(ss+1)
            loc.append(-2+np.where(np.array(All_stat3[ss]) == np.min((All_stat3[ss])))[0][0])
                
    return cam_change, loc, All_stat3

def backtrack(Bounds, PT,Base):
    All_statfn= list()
    for i in range(0,len(Bounds)):
        if Bounds[i][1] in PT:
            base2 = Base + str(Bounds[i][1]) + "/" 
            files = natsorted(os.listdir(base2))
            stat = list()
            back_len = int(Bounds[i][0][4]*100)
            x = int(Bounds[i][0][0])
            y = int(Bounds[i][0][1])
            h = int(Bounds[i][0][2]/2)
            w = int(Bounds[i][0][3]/2)

            img0 = cv2.imread(base2 + str(back_len) +".jpg")[y-h:y+h,x-w:x+w]
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

            for idx in range(np.min((26750,2*back_len)),90,-5):
                img1 = cv2.imread(base2 + str(idx) +".jpg")[y-h:y+h,x-w:x+w]
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

                stat.append(measure.compare_ssim(img0,img1,multichannel=True,win_size=3))


            for idx in range(0,len(stat)-35):
                stat[idx] = np.max((stat[idx:idx+35]))

            All_statfn.append(stat)    


    Times = {}
    count = 0
    for stat in All_statfn:
        Times[Bounds[count][1]] = 999
        count+=1

    count = 0
    for stat in All_statfn:

        if np.max((stat)) > 0.5 and np.min((stat))<0.65:
            stat = savgol_filter(stat,21,1)
            nstat = (list(reversed(stat)) -min(stat))/(max(stat)-min(stat))
            Found = check_continual(nstat,150)
            if Found:
                Times[Bounds[count][1]] = np.min((Times[Bounds[count][1]],np.min(((np.where(np.array(nstat)>=0.4)[0][0])*5/30,Times[Bounds[count][1]]))))


        count+=1
        
    return Times, All_statfn
      
    
def backtrack1(Bounds,Base):
    All_statfn= list()
    for i in range(0,len(Bounds)):
        print(i)
        base2 = Base + str(Bounds[i][1]) + "/" 
        files = natsorted(os.listdir(base2))
        stat = list()
        back_track = int(Bounds[i][0][4]*100)
        x = int(Bounds[i][0][0])
        y = int(Bounds[i][0][1])
        h = int(Bounds[i][0][2]/2)
        w = int(Bounds[i][0][3]/2)

        img0 = cv2.imread(base2 + str(back_track) +".jpg")[y-h:y+h,x-w:x+w]
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

        for idx in range(np.min((26750,2*back_track)),90,-5):
            img1 = cv2.imread(base2 + str(idx) +".jpg")[y-h:y+h,x-w:x+w]
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

            stat.append(measure.compare_ssim(img0,img1,multichannel=True,win_size=3))


        for idx in range(0,len(stat)-35):
            stat[idx] = np.max((stat[idx:idx+35]))


        All_statfn.append(stat)    
        
        
    Times = {}
    count = 0
    for stat in All_statfn:
        Times[Bounds[count][1]] = 999

        count+=1

    count = 0
    for stat in All_statfn:
        stat = reject_outliers(stat)
        if np.max((stat)) > 0.5 and np.min((stat))<0.55:
            stat = savgol_filter(stat,21,1)

            nstat = (list(reversed(stat)) -min(stat))/(max(stat)-min(stat))
            Found = check_continual(nstat,150)
            if Found:
                if (np.where(np.array(nstat)>=0.4))[0][0] != 0:
                    Times[Bounds[count][1]] = np.min((Times[Bounds[count][1]],np.min(((np.where(np.array(nstat)>=0.5)[0][0])*5/30,Times[Bounds[count][1]]))))

        count+=1
        
    return Times,All_statfn
        
                    
        
        

                    
    
        
                    
    