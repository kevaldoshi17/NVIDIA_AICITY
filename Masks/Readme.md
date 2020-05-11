-- Segmentation Maps
1. Run `python3 extract_frames.py` to extract frames from the videos with frequency set as 10.
2. Run pretrained Yolo v3 model on the original_images folder and save it as `part1.json`. To reduce complexity, we divided the task into two parts and saved it as part1.json and part2.json. 
3. Run `Seg_masks.py`. 
4. Run `python3 Masks/get_ignore_area.py`.
5. Computed Segmentation Masks can be downloaded from [here](https://drive.google.com/file/d/15mcjQx02CQ4sgJ9k4UG718wwPpoGAlS4/view?usp=sharing).
