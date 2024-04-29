#!/usr/bin/env python3

import argparse
import sys
import os
import logging.handlers
import json
import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm.notebook import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale
import matplotlib.pyplot as plt
import mne
import EntropyHub
from EntropyHub import MSEn
import utilities
import shutil
from copy import deepcopy
import json

def main(analysis_name, file_paths, channels, scales, length, frequency):
  file_path_list = []
  desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
  print("==============DESKTOP PATH==============")
  print(desktop_path)
  for each in file_paths:
    file_path_list.append(os.path.abspath(os.path.join(desktop_path, each)))

  data = None
  data2 = None
  data3 = None
  data4 = None

  Mobj = EntropyHub.MSobject('SampEn')
  fixed_channels = channels
  fun_1 = []
  fun_2 = []
  
  
  for file_path in file_path_list:
  #   ############################FUNCTION 1############################
    data = single_subject_sample_entropy_at_multiple_scales_and_complexity_index_for_single_channel(Mobj, fixed_channels, scales, frequency, length, file_path)
    fun_1.append(data)

    ############################FUNCTION 2############################
    data2 = single_subject_sample_entropy_at_multiple_scales_and_complexity_index_for_multi_channel(Mobj, fixed_channels, scales, frequency, length, file_path)
    fun_2.append(data2)

  ############################FUNCTION 3############################
  data3 = multi_subject_and_multi_channel(Mobj, fixed_channels, scales, frequency, length, file_path_list)

  ############################FUNCTION 4############################
  data4 = multi_subject_and_single_channel(Mobj, fixed_channels, scales, frequency, length, file_path_list)


  # print("================= FINAL OUTPUT =================")
  
  # print(fun_obj)
  first_function = {
    "data": fun_1
  }
  second_function = {
    "data": fun_2
    }

  fun_obj = {
            "fun1 data": first_function,
            "fun2 data": second_function,
            "fun3 data": data3,
            "fun4 data": data4
            }
  file_write_path = os.path.join(os.getcwd(), f"{analysis_name}.txt")
  f= open(file_write_path,"w+")
  f.write(json.dumps(fun_obj))
  print("here is file path")
  for each in file_paths:
    print(os.path.abspath(os.path.join(desktop_path, f"{each}")))
    os.remove(os.path.abspath(os.path.join(desktop_path, f"{each}")))
    print("File Removed!")


def get_subject_data(paths):
  # print("IN POINT 4")
  data = pd.read_csv(paths)
  return data

def get_file_names(complete_file_paths):
  file_names = []
  for each in complete_file_paths:
    file_names.append(os.path.basename(each))
  return file_names

def single_subject_sample_entropy_at_multiple_scales_and_complexity_index_for_single_channel(Mobj, channel, scale, frequency, length, file_path):
  data = get_subject_data(file_path)
  data = data.iloc[:, 1:]
  ch_names = data.columns
  # print("##################### names #####################", ch_names[0])

  sfreq = frequency
  time = np.linspace(0, data.shape[0]/sfreq, data.shape[0])

  trimmed_data_length = np.arange(0, length*sfreq) # Trim to desired length in seconds
  data = data.iloc[trimmed_data_length,:]

  target_channel = channel[0]
  scales = scale
  scales_list = np.arange(1, scales+1)

  # Initialize empty dataframe to store results
  entropy_df = pd.DataFrame(columns=["channel", "entropy", "complexity_index"])

  # Compute MSE and CI for the specified channel
  Msx, CI = EntropyHub.MSEn(data[target_channel].values, Mobj, Scales=scales)

  # Append results to dataframe
  entropy_df = entropy_df.append(pd.Series({"channel": target_channel,
                                            "entropy": Msx,
                                            "complexity_index": CI}), ignore_index=True)

  # Create subplots with two columns
  fig = make_subplots(rows=1, cols=2, subplot_titles=("MSE", "Complexity Index"))

  # Add line graph for MSE
  fig.add_trace(go.Scatter(x=scales_list, y=Msx, mode='lines+markers', line=dict(color='black'),
                           marker=dict(color='black', size=8), name='MSE'), row=1, col=1)

  # Add bar graph for Complexity Index
  fig.add_trace(go.Bar(x=[target_channel], y=[CI], marker=dict(color='blue'), name='Complexity Index'), row=1, col=2)

  # Update layout
  fig.update_layout(template='simple_white', width=1200, height=600, font=dict(size=20),
                    xaxis_title='Scales', title_text=f"Analysis for {target_channel}")

  # Set x-axis tickvals and ticktext for the bar graph
  fig.update_xaxes(tickvals=[target_channel], ticktext=[target_channel], row=1, col=2)

  # Show plot
  # fig.show()

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=scales_list, y=Msx, mode='lines+markers', line=dict(color='black'),
                           marker=dict(color='black', size=8)))
  fig.update_layout(template='simple_white', width=600, height=400, font=dict(size=20),
                    xaxis_title='Scales', yaxis_title='Sample Entropy')
  # fig.show()

  graph_data = {
        # 'Sample_Entropy': {
        #     'scales_list': scales_list.tolist(),
        #     'Msx': Msx.tolist()
        # },
        'Complexity_Index': {
            'channel': target_channel,
            'CI': CI
        }
    }

  return graph_data


def single_subject_sample_entropy_at_multiple_scales_and_complexity_index_for_multi_channel(Mobj, ch_names, scale, frequency, length, file_path):

  data = get_subject_data(file_path)
  data = data.iloc[:, 1:]
  ch_names = data.columns
  # print(ch_names)

  sfreq = frequency
#   time = np.linspace(0, data.shape[0]/sfreq, data.shape[0])

  trimmed_data_length = np.arange(0, length*sfreq) # Trim to desired length in seconds
  data = data.iloc[trimmed_data_length,:]

#   target_channel = channel
  scales = scale
  scales_list = np.arange(1, scales+1)

  mse_across_channels_df = pd.DataFrame(columns=ch_names)
  ci_df = pd.DataFrame(columns=['Channel','CI'])

  for ch in tqdm(ch_names):
      ch_data = data[ch].values

      Msx, CI = EntropyHub.MSEn(ch_data, Mobj, Scales=scales)

      mse_series = pd.Series({ch:Msx})
      mse_across_channels_df[ch] = Msx
      ci_df = ci_df.append(pd.Series({'Channel':ch, 'CI':CI}), ignore_index=True)

  fig = make_subplots(cols=2, column_widths=[0.2,0.4], horizontal_spacing=0.08)
  colorscale = 'cividis'
  colors = sample_colorscale(colorscale, np.linspace(0,1,len(ch_names)), low=0.0, high=0.9, colortype='rgb')
  mse_values = []
  for i, ch in enumerate(ch_names):
      mse_vals = mse_across_channels_df[ch].values
      mse_values.append(mse_vals)
      fig.add_trace(go.Scattergl(x=scales_list, y=mse_vals, mode='lines+markers', line=dict(color=colors[i]),
                                 marker=dict(color=colors[i], size=8), name=ch), row=1, col=1)

  mean_mse = mse_across_channels_df.mean(axis=1).values
  sem_mse = mse_across_channels_df.sem(axis=1).values
  fig.add_trace(go.Scattergl(x=scales_list, y=mean_mse, error_y=dict(type='data', array=sem_mse, visible=True),
                             mode='lines+markers', line=dict(color='brown', width=6),
                             marker=dict(color='brown', size=12), name='Average'), row=1, col=1)

  fig.add_trace(go.Bar(x=ci_df['Channel'], y=ci_df['CI'], marker=dict(color=colors, line=dict(color='black', width=1)), showlegend=False), row=1, col=2)

  fig.update_layout(template='simple_white', width=1800, height=600, font=dict(size=20),
                    xaxis_title='Scales', yaxis_title='Sample Entropy', title_text=file_path)
  fig.update_yaxes(title_text='Complexity Index', row=1, col=2)
  # fig.show()

  mse_values = [arr.tolist() for arr in mse_values]

  graph_data = {
        "Sample_Entropy": {
            "scales_list": scales_list.tolist(),
            "mean_mse": mean_mse.tolist(),
            "sem_mse": sem_mse.tolist(),
            "mse_vals": mse_values
        },
        "Complexity_Index": {
            "channel": ch_names.tolist(),
            "CI": ci_df["CI"].tolist()
        }
    }
  return graph_data

def multi_subject_and_multi_channel(Mobj, fixed_channels, scales, frequency, length, paths):
  all_data_dict = {} # initialize an empty dictionary to which each subject's data will be added
  trimmed_data_length = np.arange(0, length*frequency) # define the length of data to trim down to
  subject_list = get_file_names(paths) # get the list of subject names

  for subject_name in tqdm(paths):

    data = get_subject_data(subject_name)
    data = data.iloc[:,1:] # remove the first column because that column is the timestamps column
    data = data.iloc[trimmed_data_length,:] # trim the data to just the first 30sec
    all_data_dict[subject_name] = data # add the current subject's data to the dict


  data = get_subject_data(paths[0])
  scales_list = np.arange(1, scales+1)
  all_channels = data.columns
  chs = []
  for ch in all_channels:
    if ch not in fixed_channels:
      chs.append(ch)

  chs_to_exclude = ['VEOG', 'HEOG', 'LeftMast', 'RightMast'] # define any channels to exclude
  # chs_to_exclude = chs
  # print("==============CHANNELS TO EXCLUDE================")
  # print(chs_to_exclude)

  # initialize empty dataframes to which the data will be added in the for loop
  mse_across_subjects_df = pd.DataFrame(columns=['Subject','Scale','Entropy'])
  ci_across_subjects_df  = pd.DataFrame(columns=['Subject','CI'])

  for subject_name in tqdm(paths):
      # print("==============SUBJECT NAME================")
      # print(subject_name)
      data = all_data_dict[subject_name]
      # print("==============DATA================")
      # print(data)
      # Apply the boolean indexer to columns
      data = data.loc[:, [ch not in chs_to_exclude for ch in data.columns]]

      mean_eeg = data.mean(axis=1).values  # compute mean eeg trace
      Msx, CI = EntropyHub.MSEn(mean_eeg, Mobj, Scales=scales)  # compute entropy across scales and complexity index

      # add the current subject's data to the overall dataframes
      subject_mse_df = pd.DataFrame({'Subject':subject_name, 'Scale':scales_list, 'Entropy':Msx})
      mse_across_subjects_df = mse_across_subjects_df.append(subject_mse_df)
      ci_across_subjects_df = ci_across_subjects_df.append(pd.Series({'Subject':subject_name, 'CI':CI}), ignore_index=True)

  fig = go.Figure()
  # fig.show()
  # compute mean and standard error across subjects for MSE
  avg_mse_across_subjects = mse_across_subjects_df.groupby(['Scale']).mean().reset_index()['Entropy'].values
  sem_mse_across_subjects = mse_across_subjects_df.groupby(['Scale']).sem().reset_index()['Entropy'].values

  # plot each subject's data
  for subject_name in mse_across_subjects_df['Subject'].unique():
      subject_data = mse_across_subjects_df[mse_across_subjects_df['Subject'] == subject_name]

      fig.add_trace(go.Scattergl(x=subject_data['Scale'].values, y=subject_data['Entropy'].values, mode='lines+markers', line=dict(color='slategrey'),
                                marker=dict(color='slategrey', size=8), name=subject_name))

  # add line for average across subjects
  fig.add_trace(go.Scattergl(x=scales_list, y=avg_mse_across_subjects, error_y=dict(type='data', array=sem_mse_across_subjects, visible=True),
                            mode='lines+markers', line=dict(color='black', width=6),
                            marker=dict(color='black', size=12), name='Average'))

  fig.update_layout(template='simple_white', width=1000, height=600, font=dict(size=20),
                    xaxis_title='Scales', yaxis_title='Sample Entropy')
  # fig.show()

  graph_data = {
        'Sample_Entropy': {
            'scales_list': scales_list.tolist(),
            'avg_mse_across_subjects': avg_mse_across_subjects.tolist(),
            'sem_mse_across_subjects': sem_mse_across_subjects.tolist()
        }
        ,
        'Complexity_Index': {
            'subject': mse_across_subjects_df['Subject'].unique().tolist(),
            'CI': ci_across_subjects_df['CI'].tolist()
        }
    }
  return graph_data

def multi_subject_and_single_channel(Mobj, fixed_channels, scales, frequency, length, paths):
  # print("IN POINT 9")
  data_directory = paths
  # print("==============DATA DIRECTORY================")
  # print(data_directory)
  # Generate a list of subject names based on the CSV files in the directory
  subject_list = get_file_names(data_directory)
  # print("==============SUBJECT LIST================")
  # print(subject_list)

  # Initialize variables
  all_data_dict = {}
  # trimmed_data_length = np.arange(0, 30*sfreq)
  trimmed_data_length = np.arange(0, length*frequency)
  scales_list = np.arange(1, scales+1)
  # target_channel = 'FP2'  # Specify the target channel
  target_channel = fixed_channels[0]  # Specify the target channel


  # Loop through each subject and store their data
  for subject in tqdm(paths):
    data = get_subject_data(subject)
    data = data.iloc[:,1:]  # Remove the first column (timestamps)
    data = data.iloc[trimmed_data_length,:]  # Trim to the first 30sec
    file_name = os.path.basename(subject)
    all_data_dict[file_name] = data 

  # Initialize DataFrame for MSE and CI values
  mse_single_channel_df = pd.DataFrame(columns=['Subject', 'Scale', 'Entropy'])
  ci_single_channel_df = pd.DataFrame(columns=['Subject', 'CI'])

  # Loop through each subject and calculate MSE and CI for the target channel
  for subject_name in tqdm(subject_list):
      data = all_data_dict[subject_name]
      ch_data = data[target_channel].values  # Extract data for the target channel
      Msx, CI = EntropyHub.MSEn(ch_data, Mobj, Scales=scales)  # Compute MSE and CI

      # Store MSE values in the DataFrame
      for scale, entropy in zip(scales_list, Msx):
          mse_single_channel_df = mse_single_channel_df.append({
              'Subject': subject_name,
              'Scale': scale,
              'Entropy': entropy
          }, ignore_index=True)

      # Store CI value in the DataFrame
      ci_single_channel_df = ci_single_channel_df.append({
          'Subject': subject_name,
          'CI': CI
      }, ignore_index=True)

  # Plotting for MSE
  fig_mse = go.Figure()

  # Compute mean and standard error across subjects for MSE
  avg_mse = mse_single_channel_df.groupby(['Scale'])['Entropy'].mean().values
  sem_mse = mse_single_channel_df.groupby(['Scale'])['Entropy'].sem().values

  # Plot the data for each subject
  for subject_name in mse_single_channel_df['Subject'].unique():
      subject_data = mse_single_channel_df[mse_single_channel_df['Subject'] == subject_name]
      fig_mse.add_trace(go.Scattergl(x=subject_data['Scale'], y=subject_data['Entropy'],
                                    mode='lines+markers', name=subject_name))

  # Add line for average across subjects
  fig_mse.add_trace(go.Scattergl(x=scales_list, y=avg_mse, error_y=dict(type='data', array=sem_mse, visible=True),
                                mode='lines+markers', line=dict(color='black', width=6),
                                marker=dict(color='black', size=12), name='Average'))

  fig_mse.update_layout(template='simple_white', width=1000, height=600, font=dict(size=20),
                        xaxis_title='Scales', yaxis_title='Sample Entropy', title=f'Group-Level Single-Channel MSE for {target_channel}')
  fig_mse.show()

  # Plotting for CI (if needed)
  fig_ci = go.Figure()

  # Plot the data for CI
  for subject_name in ci_single_channel_df['Subject'].unique():
      subject_ci = ci_single_channel_df[ci_single_channel_df['Subject'] == subject_name]['CI'].values[0]
      fig_ci.add_trace(go.Bar(x=[subject_name], y=[subject_ci]))

  fig_ci.update_layout(template='simple_white', width=1000, height=600, font=dict(size=20),
                      xaxis_title='Subject', yaxis_title='Complexity Index', title=f'Group-Level Single-Channel CI for {target_channel}')
  fig_ci.show()

  graph_data = {
        'Sample_Entropy': {
            'scales_list': scales_list.tolist(),
            'avg_mse': avg_mse.tolist(),
            'sem_mse': sem_mse.tolist()
        },
        'Complexity_Index': {
            'subject': ci_single_channel_df['Subject'].tolist(),
            'CI': ci_single_channel_df['CI'].tolist()
        }
    }
  return graph_data

################################## MAIN ##################################

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='EEG Analyzer')
  
  parser.add_argument('--function', choices=['main', 'test'], default='main', dest='function', help='Function to call')
  parser.add_argument('--analysis-name', dest='analysis_name', default=None, help='Analysis Name')
  parser.add_argument('--subject-list', dest='subject_list', nargs='+', help='Subject-path-list')
  parser.add_argument('--scales', dest='scales', type=int, help='Scale')
  parser.add_argument('--length', dest='length', type=int, help='Length')
  parser.add_argument('--frequency', dest='frequency', type=int, help='Frequency')
  parser.add_argument('--channels', dest='channels', nargs='+', help='Channels')

  # Parse command-line arguments
  args = parser.parse_args()
  # Call the specified function with arguments
  if args.function == 'main':
      main(args.analysis_name, args.subject_list, args.channels, args.scales, args.length, args.frequency)

