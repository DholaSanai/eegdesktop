#!/usr/bin/env python3

import argparse
import sys
import os
import logging.handlers
import json
import pandas as pd

def test():
  print("Hello World!")
  print("Python version")
  print(sys.version)

def main(analysis_name, file_path_list, channels, scales, length, frequency):
  # print(analysis_name)
  for file_path in file_path_list:
    absolute_path = os.path.abspath(file_path)
    print(absolute_path)
    print(os.path.exists(absolute_path)) 
  # print(channels)
  print(scales)
  # print(length)
  # print(frequency)



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
  print(args.subject_list)
  # Call the specified function with arguments
  if args.function == 'main':
      main(args.analysis_name, args.subject_list, args.channels, args.scales, args.length, args.frequency)
  else:
      test()