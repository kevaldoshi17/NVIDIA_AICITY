import glob
import os
from pathlib import Path

import cv2
import numpy as np
import tqdm

root = "../Data/test-data/"
dest_dir = "ori_images/"
video_names = [str(i)+'.mp4' for i in range(1,101)]
print("capture videos")
for video_name in tqdm.tqdm(video_names):
    file_name = video_name
    folder_name = dest_dir+file_name.split('.')[0]
    os.makedirs(folder_name,exist_ok=True)
    vc = cv2.VideoCapture(root+video_name)
    c = 1
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    timeF =100
    pic_path = folder_name+'/'

    while rval: 
        if (c % timeF == 0): 
            cv2.imwrite(pic_path + str(c) + '.jpg', frame) 
        c = c + 1
        cv2.waitKey(1)
        rval, frame = vc.read()
    vc.release()

dest_dir_processed = "processed_images/"

if not os.path.isdir(dest_dir_processed):
    os.mkdir(dest_dir_processed)
print("average images")
for i in tqdm.tqdm(range(1,101)):
    video_name = str(i)
    path_file_number=glob.glob(os.path.join(dest_dir,video_name,'*.jpg')) 
    internal_frame = 4
    start_frame = 100
    video_name = str(i)
    nums_frames = len(path_file_number)
    alpha=0.1
    Path(dest_dir_processed+video_name).mkdir(exist_ok=True)

    for j in range(4,5):
        internal_frame = 100
        num_pic = int(nums_frames)
        former_im = cv2.imread(dest_dir+"%d/100.jpg"%i)
        img = cv2.imread(os.path.join(dest_dir,video_name,str(start_frame)+'.jpg'))
        for i in range(num_pic):
            # print(os.path.join(dest_dir,video_name,str(start_frame)+'.jpg'))
            # print(os.path.join(dest_dir,video_name,str(i*internal_frame+start_frame)+'.jpg'))
            now_im = cv2.imread(os.path.join(dest_dir,video_name,str(i*internal_frame+start_frame)+'.jpg'))
            if np.mean(np.abs(now_im-former_im))>5:
                img = img*(1-alpha)+now_im*alpha
                cv2.imwrite(dest_dir_processed+video_name+'/'+str(i*internal_frame+start_frame)
                        +'_'+str(j)+'.jpg',img)
            else:
                cv2.imwrite(dest_dir_processed+video_name+'/'+str(i*internal_frame+start_frame)
                        +'_'+str(j)+'.jpg',img*0)
            former_im = now_im
