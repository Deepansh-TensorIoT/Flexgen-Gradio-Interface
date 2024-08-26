import glob
from pdf2image import convert_from_path
from pathlib import Path
from os import path
#from excel_extraction.Utils.LLM import (ask_claude, ask_claude_bit_information, ask_claude_to_expand, ask_claude_to_expand_cont)
import pandas as pd
import re
import time
import json
import ast
import numpy as np
from excel_extraction.Utils import excel_llm_processing_utils as elm
from S3_utils import save_to_s3
import os

def run_llm_extraction_register(BASE_DIR, vendor_name, customization):
    # Intialize
    print("BASE_DIR:", BASE_DIR)
    CSV_DIR = str(BASE_DIR) + "/Register/"
    csv_list_ = glob.glob(CSV_DIR + '*.csv')
    files = []
    text = [] 
    CSV_BIT_DIR = str(BASE_DIR) + "/Bit/"
    csv_list_bit = glob.glob(CSV_BIT_DIR + '*.csv')
    files_bit = []
    text_bit = []  

    #Start reading csv's

    dfs = elm.process_csv(csv_list_)
    tables, titles_cleaned = elm.chunk_clean_preLLM(dfs)
    # LLM stuff
    #customization = """ coefficient can be considered as the scale""" #Clou

    results = []
    for i in range(0, len(tables)):
        try:
            print("Processing record: ", i)
            result = elm.ask_claude(tables[i], titles_cleaned[i], customization)
            results.append(result)
            time.sleep(2)
        except Exception as e:
            print("Skipping due to the error ", e)

    print("Result List has been created")        
    print(results)
    # Post process LLM Stuff
    final_df = elm.convert_llm_res_to_df(results)
    filtered_df = elm.clean_df_after_extraction(final_df)
    df_size = filtered_df.shape[0]

    # Local time has date and time
    t = time.localtime()
    # Extract the time part
    current_time = time.strftime("%m_%d_%H", t)

    #Expand the register
    parent_dir = os.path.dirname(BASE_DIR)
    output_dir = os.path.join(parent_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"Modbus_addr_{vendor_name}_{current_time}.csv"
    #file_name_xlsx = f"Modbus_addr_{vendor_name}_{current_time}.xlsx"
    path = os.path.join(output_dir, file_name)
    #excel_path = os.path.join(output_dir, file_name_xlsx)
    print("File will be save in and as:" , path)
    filtered_df.to_csv(path, index= None)
    #filered_df.to_excel(excel_path, index = None)
    df_exp = elm.get_register_to_expand(filtered_df)
    df_extracted_without_ranges = elm.drop_rows(filtered_df)
    if not df_exp.empty :
        results_expand = elm.get_expanded_registers(df_exp)
        if len(results_expand) > 0 :
            df_exp_fin = elm.convert_llm_res_to_df(results_expand)
            df_extracted_without_ranges = pd.concat([df_extracted_without_ranges, df_exp_fin], ignore_index=True)
    df_extracted_without_ranges['reg_id'] = df_extracted_without_ranges['component_name'].str.replace(" ", "_").str.lower()
    df_extracted_without_ranges['reg_id'] = df_extracted_without_ranges['reg_id'].str.replace("-", "").str.lower()
    df_extracted_without_ranges = df_extracted_without_ranges.fillna('Not Found')
    if df_extracted_without_ranges.shape[0] > df_size:
        df_extracted_without_ranges.to_csv(path, index= None)

    # Bit information processing
    df_bit_fin = pd.DataFrame()
    if len(csv_list_bit) > 0:
        responses = elm.process_bit_info(csv_list_bit)
        df_bit_fin = elm.process_llm_bit_response(responses)
        # Update the dataframes
        if df_bit_fin.size > 0:
            df_bit_fin, df_extracted_without_ranges_bit = elm.update_bit_info_with_partial_match(df_bit_fin, df_extracted_without_ranges)
    else:
        df_extracted_without_ranges_bit = df_extracted_without_ranges.copy()
            
    #Format output
    df_extracted_without_ranges_bit['register_address'] = df_extracted_without_ranges_bit['register_address'].astype(str)
    df_extracted_without_ranges_bit['component_name'] = df_extracted_without_ranges_bit['component_name'].replace(r'[^\w\s]|_', '', regex=True)
    df_extracted_without_ranges_bit['component_name'] = df_extracted_without_ranges_bit['component_name'].str.lower()
    df_extracted_without_ranges_bit = df_extracted_without_ranges_bit.replace("Not Found", '', regex=True)
    df_extracted_without_ranges_bit = df_extracted_without_ranges_bit.replace("not found", '', regex=True)
    df_extracted_without_ranges_bit = df_extracted_without_ranges_bit.replace("nan", "")
    df_extracted_without_ranges_bit = df_extracted_without_ranges_bit.replace("NaN", "")
    df_extracted_without_ranges_bit = df_extracted_without_ranges_bit.fillna('') 
    df_extracted_without_ranges_bit['reg_id'] = df_extracted_without_ranges_bit['component_name'].str.replace(" ", "_").str.lower()
    df_extracted_without_ranges_bit['reg_id'] = df_extracted_without_ranges_bit['reg_id'].str.replace('[|]|(|)|{|}|-|%|*|@|\|/', '').str.lower()

    # Apply the function to the 'register address' column
    df_extracted_without_ranges_bit['register_address'] = df_extracted_without_ranges_bit['register_address'].apply(elm.remove_hex_prefix)
    # Apply the function to the 'data type' column
    df_extracted_without_ranges_bit['data_type'] = df_extracted_without_ranges_bit['data_type'].apply(elm.standardize_data_type)
    
    print("Shape of the final dataframe: ", df_extracted_without_ranges_bit.shape[0])
    print("Saving here --> ", path )
    if df_extracted_without_ranges_bit.shape[0] >= df_size:
        df_extracted_without_ranges_bit.to_csv(path, index= None)
        #df_extracted_without_ranges_bit.to_excel(path, index = None)

        # Rearrange the columns
    column_to_move = 'sheet_name'
    cols = list(df_extracted_without_ranges_bit.columns)
    cols.remove(column_to_move)
    cols.append(column_to_move)
    df_extracted_without_ranges_bit = df_extracted_without_ranges_bit[cols]

    # Group by "sheet name"
    grouped = df_extracted_without_ranges_bit.groupby('sheet_name')
    # Write the groups to the same CSV file
    first = True
    for sheet_name, group in grouped:
        if first:
            group.to_csv(path, index=False)
            #group.to_excel(path, index = False)
            first = False
        else:
            group.to_csv(path, mode='a', header=False, index=False)
            #group.to_excel(path, mode = 'a', header = False, index = False)
            
    if not df_bit_fin.empty:
        df_bit_fin.to_csv(path, index = None, mode = 'a')
        #df_bit_fin.to_excel(path, index = None, mode = 'a')
        
    df_extracted_without_ranges_bit_complete = pd.read_csv(path)
    excel_file_name = f"Modbus_addr_{vendor_name}_{current_time}.xlsx"
    path_excel = os.path.join(output_dir, excel_file_name)
    with pd.ExcelWriter(path_excel, engine='xlsxwriter') as writer:
        df_extracted_without_ranges_bit_complete.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        text_format = workbook.add_format({'num_format': '@'})
        worksheet.set_column('A:A', None, text_format)
        
    return df_extracted_without_ranges_bit_complete

def llm_driver_function(BASE_DIR, vendor_name, model_input, version_input, customization):
    register_tables = run_llm_extraction_register(BASE_DIR, vendor_name, customization)
    parent_dir = os.path.dirname(BASE_DIR)
    output_dir = os.path.join(parent_dir, "output")
    save_to_s3(vendor_name, model_input, version_input, output_dir, folder_name = "final_output")
    print("Processing Excel File Complete")
    return register_tables