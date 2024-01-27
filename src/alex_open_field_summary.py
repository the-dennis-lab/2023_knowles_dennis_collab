"""alex_open_field_summary.py:
REQUIRED INPUTS:
    1. a file or folder of files made by alex_open_field.py _predictions.csv

OUTPUTS:
a summary file in a fresh results folder at ../data/results_{date,time} folder"""

__author__ = "ejd"
__credits__ = ["ejd"]
__maintainer__ = "ejd"
__email__ = "dennise@hhmi.org"
__license__ = "MIT"
__status__ = "Development"

import numpy as np
import pandas as pd
import glob, os, csv, sys, cv2, math, itertools, joblib, datetime, time
import seaborn as sns
if __name__ == "__main__":

    os.chdir(os.path.dirname(__file__))

    FPS=29.93
    frames_with_nose=[]
    frames_with_head=[]
    frames_with_body=[]
    l_frames=[]
    frames_within_50mm=[]
    fraction_frames_within_50mm=[]
    fraction_LIGHT_frames_within_50mm=[]
    fraction_LIGHT_frames_in_center=[]
    first_frame_in_center=[]
    num_entrances=[]
    total_frames=[]
    num_jumps=[]
    num_rears=[]
    num_frames_jumping=[]
    num_frames_rearing=[]
    first_nose_in_box=[]
    first_head_in_box=[]
    first_body_in_box=[]
    sum_dist=[]
    sum_dist_without_jumps=[]

    today = datetime.datetime.today()
    new_results_fld = '../data/results_{}'.format(today.strftime("%Y-%m-%d-%H%M%S"))

    if not os.path.isdir(new_results_fld):
        os.mkdir(new_results_fld)
    else:
        time.sleep(2) # wait 2 seconds
        today = datetime.datetime.today()
        new_results_fld = '../data/results_{}'.format(today.strftime("%Y-%m-%d-%H%M%S"))


    # arg parser
    file_or_folder=sys.argv[1]
    if os.path.isdir(file_or_folder):
        folder_of_files = file_or_folder
        folder_list = os.listdir(file_or_folder)
        full_path_list = [os.path.join(file_or_folder,file) for file in folder_list if 'predictions.csv' in file]
    elif os.path.isfile(file_or_folder):
        full_path_list = [file_or_folder]
    else:
        print('after alex_open_field.py, please provide a file or folder. You typed {} and this does not exist'.format(file_or_folder))


    sub_file_list=[]
    for filename in full_path_list:
        sub_file_name=os.path.basename(filename)[0:19]
        sub_file_list.append(sub_file_name)
        print('on {}'.format(sub_file_name))

        # load df
        df = pd.read_csv(filename,header=[0,1])
        try:
            df=df.drop(columns=['Unnamed: 0_level_0'])
        except:
            1

        nose_vals = ~np.isnan(df.nose['x'])
        ear_l_vals = ~np.isnan(df.ear_left['x'])
        ear_r_vals = ~np.isnan(df.ear_right['x'])
        tail_vals = ~np.isnan(df.tail_base['x'])

        # get some summary data: # of frames with the nose in?
        # nose + ears (aka head)? nose + ears + tailbase? (aka body)
        # I chose these points because the paws and tail tip aren't always
        # picked up as well by DLC, so these are high-fidelity pts
        frames_with_nose.append(np.sum(nose_vals))
        # if the nose entered the box, find the first entrance frame, else: nan
        if frames_with_nose[-1]>0:
            first_nose_in_box.append(np.argmax(nose_vals))
        else:
            first_nose_in_box.append(np.argmax(nose_vals))
        frames_with_head.append(np.sum(nose_vals*ear_l_vals*ear_r_vals))
        # if the HEAD entered the box, find the first entrance frame, else: nan
        if frames_with_head[-1]>0:
            first_head_in_box.append(np.argmax(nose_vals*ear_l_vals*ear_r_vals))
        else:
            first_head_in_box.append(np.nan)
        # if the BODY entered the box, find the first entrance frame, else: nan
        frames_with_body.append(np.sum(nose_vals*ear_l_vals*ear_r_vals*tail_vals))
        if frames_with_body[-1]>0:
            first_body_in_box.append(np.argmax(nose_vals*ear_l_vals*ear_r_vals*tail_vals))
        else:
            first_body_in_box.append(np.nan)
        total_frames.append(len(nose_vals))
            # let's generate some more summary data:
        nose_sum=np.sum(nose_vals)
        nose_df=df.nose
        nose_df['within50mm']=0
        for idx in nose_df.index:
            xval=nose_df.x[idx]
            yval=nose_df.y[idx]
            if xval < 50 or xval > 450 or yval < 50 or yval > 450:
                nose_df.iloc[idx,3]=1 #3 is the column we're filling with ones
                #if and only if the points are outside the box

        summed_val=np.sum(nose_df.within50mm)
        # how many frames are within 50mm of the edges of the box? (inluding
        # those outside the edges which are jumps or rears)
        frames_within_50mm.append(summed_val)
        fraction_frames_within_50mm.append(summed_val/len(nose_df))
        fraction_LIGHT_frames_in_center.append((nose_sum-summed_val)/nose_sum)
        fraction_LIGHT_frames_within_50mm.append(summed_val/nose_sum)
        sub_nose_df=nose_df[~np.isnan(nose_df.x)]
        sub_nose_df=sub_nose_df[sub_nose_df.within50mm<1]
        try:
            first_frame_in_center.append(sub_nose_df.index[0])
        except:
            first_frame_in_center.append(np.nan)
        # add 2 cols to df: one for if the animal was in box or not, and one for
        #the "bout number" which gives us # of entrances
        # also keep track of inter-bout-intervals
        df['in_box']=float(0)
        df['light_bout_num']=float(0)
        inter_light_interval=np.zeros(len(df))
        inter_light_bout=0
        bout_num=0
        for idx in df.index:
            if np.isnan(df.nose.x[idx]):
                df.iloc[idx,-2]=0
                df.iloc[idx,-1]=0
                if df.iloc[idx-1,-2]>0:
                    inter_light_bout+=1
                inter_light_interval[idx]=inter_light_bout
            elif df.iloc[idx-1,-2]==0:
                bout_num+=1
                df.iloc[idx,-2]=1
                df.iloc[idx,-1]=bout_num
            else:
                df.iloc[idx,-2]=1
                df.iloc[idx,-1]=bout_num
        num_entrances.append(np.max(df.light_bout_num)[0])
            # get the number of unique 4+ length jumps
        num_jumps.append(len(np.unique(df.jump_bouts)))
        num_rears.append(len(np.unique(df.rear_bouts)))
        num_frames_jumping.append(np.sum(~np.isnan(df.jump_bouts[df.jump_bouts>0]))[0])
        num_frames_rearing.append(np.sum(~np.isnan(df.rear_bouts[df.rear_bouts>0]))[0])
        sum_dist_without_jumps.append(np.sum(df.animal_dist_without_jumps)[0])
        sum_dist.append(np.sum(np.abs(df.animal_dist_traveled))[0])
            # make a dataframe with all summary data, save as csv
        column_vals=['sub_file_name','total_frames','num_entrances','first_frame_with_nose','first_frame_with_head','first_frame_with_body','first_frame_in_center','frames_with_nose','frames_with_head','frames_with_body','frames_within_50mm','fraction_frames_within_50mm','fraction_LIGHT_frames_within_50mm','fraction_LIGHT_frames_in_center','num_jumps','sum_dist','sum_dist_without_jumps','num_frames_jumping','num_rears','num_frames_rearing']
        zipped=zip(sub_file_list,total_frames,num_entrances,first_nose_in_box,first_head_in_box,first_body_in_box,first_frame_in_center,frames_with_nose,frames_with_head,frames_with_body,frames_within_50mm, fraction_frames_within_50mm,fraction_LIGHT_frames_within_50mm,fraction_LIGHT_frames_in_center,num_jumps,sum_dist,sum_dist_without_jumps,num_frames_jumping,num_rears,num_frames_rearing)
        summary_df=pd.DataFrame(zipped,columns=column_vals)
        summary_df.to_csv(os.path.join(new_results_fld,"{}_intermediate_summary.csv".format(today.strftime("%Y-%m-%d-%H%M%S"))))
    # save final summary with date and time
    today = datetime.datetime.today()
    summary_df.to_csv(os.path.join(new_results_fld,today.strftime("%Y-%m-%d-%H%M%S")+"_final_summary.csv"))



#
