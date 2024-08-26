from PIL import Image
import glob
from pdf2image import convert_from_path
from pathlib import Path
from os import path
import os
import pandas as pd
import shutil
import numpy as np
from pdf_processing_new.src import pdf_llm_processing_utils as plm
import logging
import time
from S3_utils import save_to_s3


logger = logging.getLogger(__name__)


def get_temp_paths(input_file_path, vendor_name):
    directory_path = os.path.dirname(input_file_path)
    dir_path = os.path.join(directory_path, f'{vendor_name}')
    os.makedirs(str(dir_path), exist_ok=True)
    logger.info(f"Working with vendor {vendor_name} at: {dir_path}")
    BASE_DIR = Path(str(dir_path))
    file_path = Path(str(directory_path))
    file_list = file_path.glob("*.pdf")
    IMAGE_PATH = os.path.join(dir_path, 'pages')
    CSV_FILES_PATH = os.path.join(dir_path, 'csv')
    CSV_PATH_TEXTRACT = os.path.join(CSV_FILES_PATH, 'Unprocessed/')
    RESULT_PATH = os.path.join(str(BASE_DIR), "result/")
    return dir_path, BASE_DIR, IMAGE_PATH, CSV_FILES_PATH, CSV_PATH_TEXTRACT, file_list, RESULT_PATH


def get_interim_data(
    vendor_name,
    input_file_path,
    start_page,
    start_page_table,
    end_page,
    appendix_start_page,
    app_table,
    customization,
    model_name,
    version_name
) -> pd.DataFrame:
    
    _, BASE_DIR, IMAGE_PATH, CSV_FILES_PATH, CSV_PATH_TEXTRACT, file_list, _ = get_temp_paths(input_file_path, vendor_name)
    shutil.rmtree(str(IMAGE_PATH), ignore_errors=True)
    shutil.rmtree(str(CSV_FILES_PATH), ignore_errors=True)
    shutil.rmtree(str(CSV_PATH_TEXTRACT), ignore_errors=True)
    
    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(CSV_FILES_PATH, exist_ok=True)
    os.makedirs(CSV_PATH_TEXTRACT, exist_ok=True)
    
    plm.convert_page_to_image(file_list,IMAGE_PATH, start_page, end_page )
    
    i = 0
    titles = {}
    
    # pass each image thru textract 
    image_list = Path(IMAGE_PATH).glob("*.jpg")
    for image in image_list:
        titles = plm.get_tables(image, CSV_PATH_TEXTRACT, titles)
    
    # Extract data from the csv
    csv_list_ = glob.glob(CSV_PATH_TEXTRACT + '*.csv')
    dfs = []
        
    dir_list = os.listdir(str(CSV_PATH_TEXTRACT)) 
    logger.info("Files and directories in : %s", dir_list)  
    
    reg_dir = str(BASE_DIR) + "/csv/Registers"
    bits_dir = str(BASE_DIR) + "/csv/Bits"
    sorted_file_paths = sorted(csv_list_, key=plm.extract_table_numbers)
    logger.info("sorted_file_paths: %s", sorted_file_paths)

    if start_page_table and start_page:
        start_index = sorted_file_paths.index(f"{CSV_PATH_TEXTRACT}Page_{start_page-1}_table_{start_page_table-1}.csv")
        sorted_file_paths = sorted_file_paths[start_index:]

    table_index, app_start_index = plm.process_csv_files(sorted_file_paths, reg_dir, bits_dir, appendix_start_page, app_table, titles)
    if app_start_index == 0:
        app_start_index = table_index
   
    save_to_s3(vendor_name, model_name, version_name, CSV_FILES_PATH, folder_name = "intermediate_output_csvs")
    return f"Pdf preprocessing completed. Intermediate files stored on S3 bucket for vendor: {vendor_name}, model: {model_name}, version: {version_name}"
    

#-----------------------------------
def extract_data_with_llm(
    vendor_name,
    input_file_path,
    customization,
    model_name,
    version_name
) -> pd.DataFrame:
    
    _, BASE_DIR, _, CSV_PATH, _, _, RESULT_PATH= get_temp_paths(input_file_path, vendor_name)

 
    reg_list = glob.glob(CSV_PATH + '/Registers/*.csv')
    print("register_list:",reg_list)
    
    # Sort the files by the numeric part of the filename
    reg_list.sort(key=plm.extract_number)
    # Create chunks for LLM 
    tables, titles_cleaned = plm.chunk_clean_preLLM(reg_list)

    # Local time has date and time
    t = time.localtime()

    print("Process Completed till here: -------------------------")

    # Extract the time part
    current_time = time.strftime("%m_%d_%H", t)
    print('Current time is ', current_time)

    
    results = []
    for i in range(0, len(tables)):
        try:
            print(f"Processing record: {i} out of {len(tables)}")
            result = plm.ask_claude(tables[i], titles_cleaned[i], customization)
            print("Result:",result)
            results.append(result)
            time.sleep(1)
        except Exception as e:
            print("Skipping due to the error ", e)


    final_df = plm.convert_llm_res_to_df(results, titles_cleaned)
    df_extracted = plm.clean_df_after_extraction(final_df)
    df_need_to_exp_cleaned = plm.get_register_to_expand(df_extracted)
    df_extracted_without_ranges = plm.drop_rows(df_extracted)
    
    print(df_extracted)
    if df_need_to_exp_cleaned.size > 0:
        results_expand = plm.get_expanded_registers(df_need_to_exp_cleaned)
#         print("Expanded Results:",results_expand)
        df_exp_fin = plm.process_exp_reg_output(results_expand)
        df_extracted_without_ranges = pd.concat([df_extracted_without_ranges, df_exp_fin], ignore_index=True)
    # # Display the final DataFrame
    df_extracted_without_ranges['bit_info'] = df_extracted_without_ranges['bit_info'].fillna('Not Found')

    CSV_BIT_DIR = str(BASE_DIR) + "/csv/Bits/"
    csv_list_bit = glob.glob(CSV_BIT_DIR + '*.csv')
    files_bit = []
    text_bit = []   

    # Bit information processing
    df_bit_fin = pd.DataFrame()
    if len(csv_list_bit) > 0:
        responses = plm.process_bit_info(csv_list_bit)
        df_bit_fin = plm.process_llm_bit_response(responses)
        # Update the dataframes
        if df_bit_fin.size > 0:
            df_bit_fin, df_extracted_without_ranges = plm.update_bit_info_with_partial_match(df_bit_fin, df_extracted_without_ranges)


    # Not found in 'register address' column is replaced by 
    # filtered_df_final['register_address'] = filtered_df_final['register_address'].apply(remove_hex_prefix)
    # Apply the function to the 'register address' column
    df_extracted_without_ranges['register_address'] = df_extracted_without_ranges['register_address'].apply(plm.remove_hex_prefix)
    # Apply the function to the 'data type' column
    df_extracted_without_ranges['data_type'] = df_extracted_without_ranges['data_type'].apply(plm.standardize_data_type)



    df_extracted_without_ranges['register_type'] = df_extracted_without_ranges['register_type'].str.lower()
    df_extracted_without_ranges['reg_id'] = df_extracted_without_ranges['component_name'].str.replace(" ", "_").str.lower()
    df_extracted_without_ranges['reg_id'] = df_extracted_without_ranges['reg_id'].str.replace('[|]|(|)|{|}|-|%|*|@|\|/', '').str.lower()
    df_extracted_without_ranges = df_extracted_without_ranges.replace("Not Found", '', regex=True)
    df_extracted_without_ranges = df_extracted_without_ranges.replace("not found", '', regex=True)
    df_extracted_without_ranges = df_extracted_without_ranges.replace("nan", "")
    df_extracted_without_ranges = df_extracted_without_ranges.fillna('') 
    df_extracted_without_ranges = df_extracted_without_ranges.replace('None', '')
    print("Shape before drop duplicates: ", df_extracted_without_ranges.shape[0])
    df_extracted_without_ranges = df_extracted_without_ranges.drop_duplicates(keep='first')
    print("Shape after drop duplicates: ", df_extracted_without_ranges.shape[0])


    if "CATL" in vendor_name:
        try: 
            df_extracted_without_ranges = df_extracted_without_ranges.groupby(['register_address']).agg({
            'data_type': lambda x: next(iter(x.dropna()), np.nan),  
            'word_count': lambda x: next(iter(x.dropna()), np.nan), 
            'register_type': lambda x: next(iter(x.dropna()), np.nan),
            'scale_factor': lambda x: next(iter(x.dropna()), np.nan), 
            'unit': lambda x: next(iter(x.dropna()), np.nan),  
            'bit_info': lambda x: ' '.join(x.dropna()),   # Concatenate non-null 'bit info' values
            'reg_id': lambda x: next(iter(x.dropna()), np.nan), 
    #         'component_name': lambda x: next(iter(x.dropna()), np.nan),
            }).reset_index()
        
        except Exception as e:
            print("Error Aggregating: ", e)
   
    os.makedirs(RESULT_PATH, exist_ok=True)
    path = str(RESULT_PATH) + 'Modbus_addr_'+ vendor_name + '_' + current_time + '.csv'
    print(path)
    df_extracted_without_ranges.to_csv(path, index= None)
    save_to_s3(vendor_name, model_name, version_name, RESULT_PATH, folder_name = "final_output")

    if not df_bit_fin.empty:
        mask = df_bit_fin.applymap(lambda x: str(x).lower() == "not found")
        # Drop rows where any cell matches the mask
        df_bit_fin = df_bit_fin[~mask.any(axis=1)]
        df_bit_fin.to_csv(path, index = None, mode = 'a')
        
    df = pd.read_csv(path)
    path_xl = path.replace(".csv", ".xlsx")
    with pd.ExcelWriter(path_xl, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        text_format = workbook.add_format({'num_format': '@'})
        worksheet.set_column('A:A', None, text_format)
    save_to_s3(vendor_name, model_name, version_name, path_xl, folder_name = "final_output")

    return df