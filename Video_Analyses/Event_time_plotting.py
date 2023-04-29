#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 07:45:34 2019

@author: bradly
"""
#import Libraries
# Built-in Python libraries
import os # functions for interacting w operating system

# 3rd-party libraries
import pandas as pd
import numpy as np # module for low-level scientific computing
import easygui
from datetime import datetime
from matplotlib.patches import Rectangle
from collections import OrderedDict
import matplotlib.pyplot as plt
from re import search
import matplotlib as mpl
import matplotlib.ticker as tkr  

# =============================================================================
# #Define functions
# =============================================================================
def to_xy_stone(some_dictionary):
    x, y = [], []
    for i,event in some_dictionary.items():
        y.extend([i]*len(event))
        x.extend(event)
    x, y = np.array(x).astype(np.int64), np.array(y)
    return x,y
	
def plot_events_stone(xval,yval,labs,vid_info):
	#set color palette
	#colors = plt.get_cmap('winter_r')(np.linspace(0, 1, len(labs)))
	
	fig, ax = plt.subplots(figsize=(24, 16))
	ax.set_xlim([0, np.max(xval)])
	ax.set_ylim([-0.5, np.max(yval)+0.5])
	for i in range(0, len(yval), 2):
		ax.add_patch(Rectangle((xval[i], yval[i]-0.25), xval[i+1] - xval[i], 0.5, facecolor = 'blue'))
	plt.yticks(range(yval.max()+1),labs);
	
	#Formatting
	mpl.rcParams['ytick.labelsize'] = 20
	mpl.rcParams['xtick.labelsize'] = 20
	plt.xticks(range(0,1200001,60000))
	ax.get_xaxis().set_major_formatter(tkr.FuncFormatter(numfmt))
	plt.ylabel('Event',size=21,fontweight="bold")
	plt.xlabel('Time (minute)',size=21,fontweight="bold")
	
	session = search('_(.+)_', vid_info[-15:])[0]
	plt.title('Animal: %s \nCondition: %s' %(vid_info[:4], session.replace("_","")),size=24,fontweight="bold")  
	
	#save output
	dirname = os.getcwd()
	fig.savefig(dirname+'/%s_%s' %(vid_info[:4], session.replace("_",""))+ '_Events.png')   
	plt.close(fig)	
	
def numfmt(x, pos): 
    s = '{}'.format(x / 60000.0)
    return s
  
# =============================================================================
# Import/Open HDF5 File and variables
# =============================================================================

#Get name of directory where the data files and xlsx file sits, and change to that directory for processing
dir_name = easygui.diropenbox()
os.chdir(dir_name)

#Look for the xlsx file in the directory
file_list = os.listdir('./')
excel_name = ''
for files in file_list:
    if files[-4:] == 'xlsx':
        excel_name = files

#Open file
xl_file = pd.ExcelFile(excel_name)

dfs = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names}

#Flip through dataframes to plot
for i in range(len(dfs)):  #Flip through all sheets
	#Get sheet name
	sheet = next(iter(dfs))
	
	#Locate start times (00:00:0001) <--Scored this way to allow for pandas processing
	sheet_df = dfs[sheet].reset_index()
	start_rows = sheet_df[sheet_df['Video_File_Name'].str.contains('BS',regex=False)==True]
	
	#Establish Video names in file
	video_names = list(np.array(start_rows['Video_File_Name']))
	
	newStart = []; newEnd =[]; timediff= []
	for n in range(len(sheet_df)):
		#convert and calculate time differences
		raw_start = datetime.strptime(str(sheet_df['Time_Start_(min:sec.ms)'][n]),"%H:%M:%S.%f")
		raw_end = datetime.strptime(str(sheet_df['Time_End'][n]),"%H:%M:%S.%f")
		raw_diff = datetime.strptime(str(sheet_df['Time_End'][n]),"%H:%M:%S.%f")-datetime.strptime(str(sheet_df['Time_Start_(min:sec.ms)'][n]),"%H:%M:%S.%f")
		
		#store time values
		newStart.append((raw_start.minute)*60000+(raw_start.second)*1000+raw_start.microsecond/1000)
		newEnd.append((raw_end.minute)*60000+(raw_end.second)*1000+raw_end.microsecond/1000)
		timediff.append((raw_diff.seconds)*1000+raw_diff.microseconds/1000)
		
	sheet_df['Start_time_ms'] = newStart
	sheet_df['End_time_ms'] = newEnd
	sheet_df['Time_diff_ms'] = timediff
	
	QFreq=[] #For storing frequency of motion arrays	
	for session in range(len(start_rows)): #Flip through video sessions
		if session==0:
			#set session query
			sess_query = sheet_df[start_rows.index[session]:start_rows.index[session+1]]
		else: 
			#set session query
			sess_query = sheet_df[start_rows.index[session]:sheet_df.iloc[-1,:][0]+1]
				
		#start time
		timer = int(sess_query['Start_time_ms'][start_rows.index[session]])
		
		#reset time
		sess_query['Start_time_ms'] =  sess_query['Start_time_ms']-timer
		sess_query['End_time_ms'] =  sess_query['End_time_ms']-timer
		 
		#Create categorical values for events and recode
		sess_query.Event_1 = pd.Categorical(sess_query.Event_1)
		sess_query['code'] = sess_query.Event_1.cat.codes
	
		#Extract events
		events = sess_query['Event_1'].unique()
		event_codes = sess_query['code'].unique()
		
		event_vals = []
		for event in range(len(sess_query['Event_1'].unique())):
			event_extract = sess_query['Event_1'].unique()[event]
			newQ = sess_query.query('Event_1 == @event_extract')
			
			#Store event times as vectors in list
			event_vals.append(np.array(newQ[['Start_time_ms','End_time_ms']]).flatten())
			
		#Create dictionary using event names as keys
		event_dict = dict(zip(events,event_vals))
		event_dict_coded = dict(zip(event_codes,event_vals))
		beta = OrderedDict(sorted(event_dict_coded.items(), key=lambda x: x[0]))
		alpha = OrderedDict(sorted(event_dict.items(), key=lambda x: x[0]))

		#Plot data
		x,y = to_xy_stone(beta)
		plot_events_stone(x,y,np.array([*alpha]),video_names[session])

# =============================================================================
# 		#Start frequency analyses for movement	
# =============================================================================
		motion_dict = {k: event_dict[k] for k in ('Q1', 'Q2', 'Q3', 'Q4', 'Rearing')}
		codes = [0,1,2,3,4]
		motion_dict_coded= dict(zip(codes, list(motion_dict.values())))
		x,y = to_xy_stone(OrderedDict(sorted(motion_dict_coded.items(), key=lambda x: x[0])))
 		
		#Restrict to start times (eliminates double counts)
		start_x, start_y = x[::2], y[::2]
		
		#Remove rearing from analysis
		start_xy=list(zip(*[sorted(zip(list(start_y[np.where(start_y != 4)]),list(start_x[np.where(start_y != 4)])), key=lambda x: x[1])]))
		
		#Establish where events took place
		uniquestartValues, startindicesList = np.unique([int(x[0][1]/60000) for x in start_xy], return_index=True)
		minutesnew = [int(x[0][1]/60000) for x in start_xy]
		
		#Create vector over full period
		events_by_minute = []
		for min_count in range(0,21):
			events_by_minute.append(minutesnew.count(min_count))
		
		#Store array
		QFreq.append(np.cumsum(np.array(events_by_minute)))
		
		#Plot output
		fig, ax = plt.subplots(figsize=(12,8))
		plt.plot(np.cumsum(np.array(events_by_minute)))
		
		#Formatting
		plt.ylabel('Cummulative Frequency',size=18,fontweight="bold")
		plt.xlabel('Time (minute)',size=18,fontweight="bold")
		ax.set_xlim([0, 20])
		ax.set_xticks(range(0,21,2))
		
		sessionname = search('_(.+)_', video_names[session][-15:])[0]
		plt.title('Frequency of Quadrant Movement \nAnimal: %s; Condition: %s' %(start_rows['Video_File_Name'][0][:4],sessionname.replace("_","")),size=18,fontweight="bold")  
		fig.savefig(dir_name+'/%s_ %s' %(start_rows['Video_File_Name'][0][:4], sessionname.replace("_",""))+ '_QFrequency.png')   
		plt.close(fig)
		
	#Plot QFreq together
	fig, ax = plt.subplots(figsize=(12,8))
	plt.plot(QFreq[0],lw=3,color='b',alpha=0.8)
	plt.plot(QFreq[1],lw=3,color='b',alpha=0.4)
	
	#Get video info
	vid1 = search('_(.+)_', video_names[0][-15:])[0]; vid2 = search('_(.+)_', video_names[1][-15:])[0]
	conditions = [vid1.replace("_",""), vid2.replace("_","")]
	
	#Formatting
	plt.ylabel('Cummulative Frequency',size=18,fontweight="bold")
	plt.xlabel('Time (minute)',size=18,fontweight="bold")
	plt.legend(conditions,prop={'size': 16})
	ax.set_xlim([0, 20])
	ax.set_xticks(range(0,21,2))
	plt.title('Frequency of Quadrant Movement \nAnimal: %s' %(start_rows['Video_File_Name'][0][:4]),size=18,fontweight="bold")  
	fig.savefig(dir_name+'/%s' %(start_rows['Video_File_Name'][0][:4])+ '_QFrequencyCombined.png')   
	plt.close(fig)
			









