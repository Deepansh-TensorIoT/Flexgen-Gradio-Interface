from PIL import Image
import glob
from pdf2image import convert_from_path
from pathlib import Path
from os import path
import os
import pandas as pd
import re
import shutil
import ast
import numpy as np
from pdfreader import PDFDocument
import json
# import webbrowser, os
import json
import boto3
import io
from io import BytesIO
import sys
from pdf_processing_new.src.app_logger import logger
from pathlib import Path
from collections import Counter
import time
from pdf_processing_new.src.app_logger import logger
from pprint import pprint
from anthropic import AnthropicBedrock
client = AnthropicBedrock()
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

type_mapping = {
    'u16': 'uint16',
    'i16': 'int16',
    'U16': 'uint16',
    'I16': 'int16',
    'Int16': 'int16',
    'UINT16':'uint16',
    'INT16': 'int16',
    'Uint16': 'uint16',
    'UINT32': 'uint32',
    'INT32': 'int32',
    'u32': 'uint32',
    'i32': 'int32',
    'U32': 'uint32',
    'I32': 'int32',
    's16': 'int16',
    'S16': 'int16',
    's32': 'int32',
    'S32': 'int32',
    'BITMAP': 'bitmap',
    'BITFIELD' : 'bitfield',
}


def is_page_in_range(page_num, start_page, end_page):
    return start_page <= page_num <= end_page  

def pdf_page_to_image(file_list, IMAGE_PATH):
    pages = 0
    # convert the pages of the pdf file into images
    for file in file_list:
        with open(file, 'rb') as pdf:
            doc = PDFDocument(pdf)
            pages = len(list(doc.pages()))
            logger.info("Number of pages: %s", pages)

        if start_page == None:
            logger.info("Setting start page")
            start_page = 1
        if end_page == None:
            logger.info("Setting end page")
            end_page = pages
        stem = file.stem
        logger.info("File Name: %s", stem)
        pages = convert_from_path(file)
        #Saving pages in jpeg format
        for count, page in enumerate(pages):
            Path(IMAGE_PATH).mkdir(parents=True, exist_ok=True)
            if is_page_in_range(count, start_page-1, end_page-1):
                page.save(f'{IMAGE_PATH}/Page_{count}.jpg', 'JPEG')
              
            
def is_number_or_hex(input_string):
#     hex_pattern = re.compile(r'^[0-9A-Fa-f]+$')
#     hex_pattern = re.compile(r'^(0[xX])?[0-9A-Fa-f]+$')
    contains_hex_pattern = re.compile(r'[0-9A-Fa-f]')
    numeric_pattern = re.compile(r'^\d+$')
    
    if contains_hex_pattern.match(input_string):
        return True
    elif numeric_pattern.match(input_string):
        return True
    else:
        return False
def has_more_numbers_than_text(input_string):
    num_count = sum(c.isdigit() for c in input_string)
    text_count = len(input_string) - num_count

    return num_count > text_count

def convert_page_to_image(file_list,IMAGE_PATH, start_page, end_page):
    pages = 0
    # convert the pages of the pdf file into images
    print("Start Page: ", start_page)
    print("End Page: ", end_page)
    print("End page frim PDF_extraction_main_driver : ", end_page)

    for file in file_list:
        with open(file, 'rb') as pdf:
            doc = PDFDocument(pdf)
            pages = len(list(doc.pages()))
            logger.info("Number of pages: %s", pages)

        if start_page == None:
            logger.info("Setting start page")
            start_page = 1
        if end_page == None:
            logger.info("Setting end page")
            end_page = pages
        stem = file.stem
        logger.info(stem)
        pages = convert_from_path(file)
        
        #Saving pages in jpeg format
        for count, page in enumerate(pages):
            Path(IMAGE_PATH).mkdir(parents=True, exist_ok=True)
            if is_page_in_range(count, start_page-1, end_page-1):
                page.save(f'{IMAGE_PATH}/Page_{count}.jpg', 'JPEG')

def get_rows_columns_map(table_result, blocks_map):
    rows = {}
    scores = []
    title = ""
    mcells = ""
    table_contains_header = False
    header_count = 0

    for relationship in table_result['Relationships']:
        for id in relationship['Ids']:
            if 'EntityTypes' in blocks_map[id] and 'COLUMN_HEADER' in blocks_map[id]['EntityTypes']:
                header_count += 1
                if header_count >= 3:
                    table_contains_header = True
                    
    for relationship in table_result['Relationships']:
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'CELL':
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    if row_index not in rows:
                        # create new row
                        rows[row_index] = {}
                    # get confidence score
                    scores.append(str(cell['Confidence']))
                     # get the text value
                    rows[row_index][col_index] = get_text(cell, blocks_map)
                    
                    
        elif relationship['Type'] == 'TABLE_TITLE':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                # get the text value
                title = title + get_text(cell, blocks_map)
                logger.info("Title only: %s", title)

        elif relationship['Type'] == 'MERGED_CELL':
            text_merged = ""
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'MERGED_CELL':
                    for relation in cell['Relationships']:
                        # Iterate through the IDs in the Relationships
                        for id_ in relation['Ids']:
                            cell_child = blocks_map[id_]
                            if "Relationships" in cell_child:
                                for item in cell_child['Relationships']:
                                    # Access the list of IDs within each dictionary using the 'Ids' key
                                    ids_list = item['Ids']
                                    # Iterate through the IDs in the list
                                    text_merged = ""
                                    for text_id in ids_list:
                                        cell_text = blocks_map[text_id]
                                        text_merged = text_merged+ " " + cell_text['Text']
                        for id_ in relation['Ids']:
                            cell_child = blocks_map[id_]
                            row_index = cell_child['RowIndex']
                            col_index = cell_child['ColumnIndex']
                            if row_index not in rows:
                                # create new row
                                rows[row_index] = {}
                                 # get the text value
                            rows[row_index][col_index] = text_merged
                        
    return rows, scores, title, table_contains_header


def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        if "," in word['Text'] and word['Text'].replace(",", "").isnumeric():
                            text += '"' + word['Text'] + '"' + ' '
                        else:
                            text += word['Text'] + ' '
    return text

def get_prev_text(textract_response, search_text):

    # Load the Textract response if it's a JSON string
    if isinstance(textract_response, str):
        textract_response = json.loads(textract_response)
    
    # Iterate through the blocks in the Textract response
    prev_line = ""
    for block in textract_response.get('Blocks', []):
        if block['BlockType'] == 'LINE':
            # Check if the block contains the search text
            if block['Text'] in search_text:
                logger.info("Has more number than text: %s", has_more_numbers_than_text(prev_line))
                if not has_more_numbers_than_text(prev_line):
                    text  = prev_line + ". " + search_text
                    return text
                else: 
                    return search_text
            prev_line = block['Text']
    # Return None if the text is not found
    return search_text

def get_table_csv_results(file, CSV_PATH, titles):
    # Calls textract
    session = boto3.Session()
    client = session.client('textract', region_name='us-east-1')
    
    with open(file, "rb") as pdf_file:
        img_test = pdf_file.read()
        response = client.analyze_document(
            Document={"Bytes": img_test}, FeatureTypes=["TABLES"]
        )
    
    
    # Get the text blocks
    blocks=response['Blocks']
#     pprint(blocks)

    blocks_map = {}
    table_blocks = []
    for block in blocks:
        blocks_map[block['Id']] = block
        if block['BlockType'] == "TABLE":
            table_blocks.append(block)

    if len(table_blocks) <= 0:
        return titles

    csv = ''
    for index, table in enumerate(table_blocks):
#         print("Enumerate table blocks: ",index) 
        title, output_file, table_contains_header = generate_table_csv(table, blocks_map, index +1, file, CSV_PATH)
        
        title = get_prev_text(response, title)
        logger.info("Title: %s", title)
        titles[output_file] = (title, table_contains_header)
        
    return titles

def generate_table_csv(table_result, blocks_map, table_index, file, CSV_PATH):
    rows, scores, title, table_contains_header = get_rows_columns_map(table_result, blocks_map)
    table_id = 'Table_' + str(table_index)
    
    # get cells.
    csv = ''
    for row_index, cols in rows.items():
        for col_index, text in cols.items():
            col_indices = len(cols.items())
            csv += '{}'.format(text) + '\t'
        csv += '\n'
        
    dirname = os.path.dirname(file)
    output_file = CSV_PATH + Path(file).stem +  "_table_" + str(table_index-1) + ".csv"
    with open(output_file, "wt") as fout:
        fout.write(csv)
    return title, output_file, table_contains_header

def get_tables(file_name, CSV_PATH, titles):
    logger.info("File Name: %s", file_name)
    titles = get_table_csv_results(file_name, CSV_PATH, titles)
    return titles

def extract_table_numbers(file_path):
    match = re.search(r'(\d+)_table_(\d+)', file_path)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0  # Return (0, 0) if no match is found
   
def has_mostly_unique_values(df):
    first_row = df.iloc[0].tolist()
    values = [str(cell).strip().lower() for cell in first_row if cell is not None and str(cell).lower() != 'nan']
    unique_values = set(values)
    return sum(values.count(value) for value in unique_values) - len(unique_values) < 1


def is_first_row_empty(df):
    first_row = df.iloc[0].tolist()
    # Count the number of empty elements in the list
    empty_count = sum(1 for elem in first_row if str(elem).lower() == 'nan')
    if empty_count > 1 :
        return True
    else:
        return False   
    
def get_header_row(df):
    # Check the first 5 rows
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        # Check if the row has more than one NaN
        if row.isna().sum() > 1:
            continue
        # Check if the row has mostly unique values
        if len(row) - len(row.drop_duplicates()) <= 1:
            return i
    return 0

def process_csv_files(file_list, reg_dir, bits_dir, appendix_start_page, app_table, titles):
    """Identifies unique tables and saves them as csv"""
    app_start_index = 0
    use_title = "No Title"
    contains_header = True
    title = ""
    found_bit_files = False
    print("Reg Dir: ", reg_dir)
    print("Bits Dir: ", bits_dir)
    
    if not os.path.exists(reg_dir):
        os.makedirs(reg_dir)
    if not os.path.exists(bits_dir):
        os.makedirs(bits_dir)
    table_index = 1
    
    current_df = pd.DataFrame()
    
    for file in file_list:
        logger.info("="*80)
        logger.info("Processing file: %s ",file)
        output_file = os.path.join(reg_dir, f'table_{table_index}.csv')
        logger.info("output_file: %s ", output_file)
        if appendix_start_page:
            app_csv = "Page_"+ str(appendix_start_page-1) + "_table_" + str(app_table-1)+".csv"
            if app_csv in file:
                app_start_index = table_index
                logger.info("app_start_index: %s ", app_start_index)
                found_bit_files = True
        if found_bit_files:
            output_file = os.path.join(bits_dir, f'table_{table_index}.csv')
        df = pd.read_csv(file, sep = '\t', header= None)
        df = df.dropna(axis=1, how='all')
        print(df)
        title, contains_header = titles[file] 
        logger.info("Title: %s", title)
        logger.info("Contain header: %s", contains_header)
    
        # If it contains header then probably new table and 
        if contains_header and not is_first_row_empty(df) :
            logger.info("Contains header and the first row is not empty, maybe a new table")
            header_row = get_header_row(current_df)
            logger.info("Header Row: %s", header_row)
            if not current_df.empty:
                current_df.columns = current_df.iloc[header_row]
                current_df = current_df.loc[header_row+1:]
                if table_index  == app_start_index:
                    output_file = os.path.join(reg_dir, f'table_{table_index}.csv')
                logger.info("Title: %s", use_title)
                current_df['Title'] = use_title  
#                 display(current_df)
                logger.info("Saving File here in inside for loop: %s",output_file )
                current_df.to_csv(output_file, index=False)
                table_index = table_index + 1
                current_df = pd.DataFrame()
            if title != "":
                use_title = title
            current_df = df
            
        elif not contains_header:
            logger.info("Table does not have header, probably a continuation ")
            current_df = pd.concat([current_df, df], ignore_index=True)

        else:
            logger.info("The first row is empty")
#             display(df)
            df = df.iloc[1:]
            current_df = pd.concat([current_df, df], ignore_index=True)

    if not current_df.empty:
        header_row = get_header_row(current_df)
        current_df.columns = current_df.iloc[header_row]
        current_df = current_df.loc[header_row+1:]
        current_df['Title'] = use_title
        if not found_bit_files:
            output_file = os.path.join(reg_dir, f'table_{table_index}.csv')
        else: 
            output_file = os.path.join(bits_dir, f'table_{table_index}.csv')
        logger.info("Saving File here: %s",output_file )
        logger.info("Last Title: %s", use_title)
        logger.info("Table index: %s", table_index)
        
        current_df.to_csv(output_file, index=False)
    
    dir_list = os.listdir(str(reg_dir)) 
    print("Files and directories in : ", dir_list)  
    return table_index, app_start_index

def chunk_clean_preLLM(reg_list):
    headers = [] # holds the headers
    prev_header = None
    tables = []
    text_header = ""
    titles_cleaned = []
    count = 0
    for file in reg_list:
        logger.info("Processing File: %s", file)
        df = pd.read_csv(file)
        print("Processing record: ", count)
        df = df.dropna(how='all', axis=1)
        print("Dataframe before spliting")

        for column in df.columns:
            df[column] = df[column].astype(str) + '\t'
        df['\n'] = '\n' 

        text_header = ""
        for i in range(0,len(df.columns)):
            text_header =  text_header + str(df.columns[i]) + '\t'
        text_header = text_header + '\n'

        if len(df) > 15:
            logger.info("+++++++Split the dataframe++++++++++++++")
            
            # Number of rows for each split
            rows_per_split = 15

            # Calculate the total number of splits needed
            total_splits = len(df) // rows_per_split + (1 if len(df) % rows_per_split != 0 else 0)

            for j in range(total_splits):
                start_index = j * rows_per_split
                end_index = min((j + 1) * rows_per_split, len(df))
                df_title = df.iloc[0]['Title']
                if df_title:
                    titles_cleaned.append(df.iloc[0]['Title'])
                split_df = df.iloc[start_index:end_index].copy()
                split_df = split_df.iloc[:,:-2]
                dfAsString = split_df.to_string(header=False, index=False)
                dfAsString = text_header + dfAsString
                res = re.sub(' +', ' ', dfAsString)
                res  = re.sub('\"', '', res)
                res  = re.sub(':', '-', res)
                res  = re.sub('~', '-', res)
                tables.append(res)
        else:
            logger.info("Small dataframe")
            df_title = df.iloc[0]['Title']
            if df_title:
                titles_cleaned.append(df.iloc[0]['Title'])
            df = df.iloc[:,:-2]
            dfAsString = df.to_string(header=False, index=False)
            dfAsString = text_header + dfAsString
            res = re.sub(' +', ' ', dfAsString)
            res  = re.sub('\"', '', res)
            res  = re.sub(':', '-', res)
            res  = re.sub('~', '-', res)
            tables.append(res)
        count = count + 1
    return tables, titles_cleaned

def preprocess_llm_input(dfs_copy):
    headers = [] # holds the headers
    prev_header = None
    tables = []
    text_header = ""
    titles_cleaned = []
    for i in range(0, len(dfs_copy)):
        logger.debug("\n")
        logger.debug("*"*50)
        logger.info("Processing i: %s", i)

        ffiltered_df = dfs_copy[i].copy()
        ffiltered_df = ffiltered_df.dropna(how='all', axis=1)
        for column in ffiltered_df.columns:
            ffiltered_df[column] = ffiltered_df[column].astype(str) + '\t'
        ffiltered_df['\n'] = '\n' 

        text_header = ""
        for i in range(0,len(ffiltered_df.columns)):
            text_header =  text_header + str(ffiltered_df.columns[i]) + '\t'
        text_header = text_header + '\n'

        if len(ffiltered_df) > 10:
            logger.info("+++++++Split the dataframe++++++++++++++")
            # Number of rows for each split
            rows_per_split = 10

            # Calculate the total number of splits needed
            total_splits = len(ffiltered_df) // rows_per_split + (1 if len(ffiltered_df) % rows_per_split != 0 else 0)

            for j in range(total_splits):
                start_index = j * rows_per_split
                end_index = min((j + 1) * rows_per_split, len(ffiltered_df))
                titles_cleaned.append(ffiltered_df.iloc[0]['Title'])
                split_df = ffiltered_df.iloc[start_index:end_index].copy()
                split_df = split_df.iloc[:,:-2]
                dfAsString = split_df.to_string(header=False, index=False)
                dfAsString = text_header + dfAsString
                res = re.sub(' +', ' ', dfAsString)
                res  = re.sub('\"', '', res)
                res  = re.sub(':', '', res)
                res  = re.sub('~', '-', res)
                tables.append(res)
        else:
            logger.info("Small dataframe")
            titles_cleaned.append(ffiltered_df.iloc[0]['Title'])
            ffiltered_df = ffiltered_df.iloc[:,:-2]
            dfAsString = ffiltered_df.to_string(header=False, index=False)
            dfAsString = text_header + dfAsString
            res = re.sub(' +', ' ', dfAsString)
            res  = re.sub('\"', '', res)
            res  = re.sub(':', '', res)
            res  = re.sub('~', '-', res)
            tables.append(res)
        return tables


def ask_claude(text_llm, title, customization):
    prompt = f"""You will be acting as an expert electronic engineer for the  company Flexgen, who is going to extract the modbus register address and other details associated with register like register type, data type that the register holds and other details that are given below.
    Your goal is to extract the following from the given text.
    - Modbus register address
    - Data type of the register
    - Word count
    - Register type
    - Bit info of what each register bit depicts
    - Scale factor
    - Unit of the Component which the address represents
    - Component name of the component whose register address is extracted
  
    You will return a list of dictionaries with the keys of the dictionary being the register address, data type of the registers, word count, the register type, the bit info of what each register bit depicts, scale factor, unit and the component name of the component whose register address is extracted.
    You should stict to extracting only these information from the text. There is also a title to the table. It could contain information of the register type(input register or holding register). Use the customization as an guideline to generate the final output. Here is the guidance document you should use as a reference to identyfying and extracting the register address: 
    <guide>
    The given text is a table. The cells are seperated by '\t' and '\n' represents the end of a given row. The first row is the header of the table. Extract the register address, data type, word count, bit information, scale factor, unit and component name and return a list of dictionaries with seven keys. The keys are register_address, data_type, register_type, bit_info, scale_factor, unit and component_name. Carefully understand the criteria for the extraction of Modbus register and the associated information.
    Register addresses: Look at the header to identify the register address. The register address can be in Decimal or Hexadecimal. Hexadecimal address usually have '0x' in their address. If there are 2 address columns, it will be identified as Dec or Hex. If the text contains addresses in decimal(dec/DEC), then always extract the decimal address. If there are no decimal addresses, but there are hexadecimal(hex/HEX) address. then extract the hexadecimal address. If using hexadecimal address add ‘0x’ to the extracted address. Do NOT extract hexadecimal addresses when decimal address are present. Register address will not be made up of english words. If there is no identifiable register address, return None. Register address cannot be called BIT or have english words. They can have characters, if they are hexadecimal addresses. There can also be range of valid address, in that case extract the entire range of address. If there is a range of address extract the entire range( first and last address). For parameters with multiple entries (like Pack1 to Pack64 or Rack 1 to Rack 64), provide the full range from the first to the last address. When there are ranges the word count will be for the entire ranges. Always put the number next to the rack/pack/cluster and then the actual component name. There can also be range of valid address, Identify and extract register ranges, which may be presented in two ways:
      a. Explicitly stated ranges (e.g., "0x0001-0x0015 will store software version")
      b. Consecutive rows that represent a range (e.g., "0x0001 Software version 1" followed by "0x0015 Software version 15")
    For consecutive rows, only combine them into a range if they meet ALL of the following criteria:
      a. They appear to be part of the same logical group (e.g., software versions or cluster information)
      b. The range covers a significant number of registers (more than 5)
      c. They are not already listed as individual entries in the text
    Do NOT combine registers into ranges in the following cases:
      a. When two consecutive registers have the same description (e.g.,"22129 System alarm status" and "22130 System alarm status") then do not make it a range. If more than 5 register address are there for a component it can be a range.
      b. When consecutive registers have different descriptions
      c. When it's clear that each register represents a distinct piece of information, even if they are related
      d. When a component uses two register, since it is need more address space. This is different from range.For eg - System alarm status can be found it two consequitive registers(22129 and 22130). This is not a range, but infact two register used to storing one component.
    For ranges, include the start and end register addresses and a description of what the range represents.
    If there's any ambiguity about whether to combine registers into a range, keep them as separate entries.
    Data Type: This is the type of data that is stored in the register. It can be either directly specied in the text as unsigned or singed int(U16 or I16 or U32) or the bytes representation can be given. Use the hints in the customization to correctly extract the value. If the data type is given Byte Quantity, then check the number of register address it is applicable for and then calculate the byte quantity for each register.
    Some of the commonly present data types are INT16, UINT16, ENUM, BOOL, BITMAP. There could be more. 
    Sometime the data type can be a bit. This is true for Discrete Input register. If the bit information is limited only to bit 0 and bit 1, then the data type could be bit.
    Word Count: 1 word = 16 bits = 2 bytes.The word count field represents the number of 16-bit registers requested to be read or written. 
    For a given register, the word count can be calculated if the data type is given or byte quantity.
    - INT16/I16/S16, 16-bit signed integer or 2 bytes, so word count will be 1
    - UINT16/U16/, 16-bit unsigned integer or 2 bytes, so word count will be 1
    - INT32/S32/I32, 32-bit signed integer or 4 bytes, so word count will be 2
    - UINT32/U32,32-bit unsigned integer or 4 bytes, so word count will be 2
    - INT64/S64/I64, 64-bit signed integer or 8 bytes, so word count will be 4
    - UINT64/U64, 64-bit unsigned integer or 8 bytes, so word count will be 4
    - Float32/F32, 32-bit float value or 4 bytes, so word count will be 2
    - If the data type is U16*8. 
    - In some case you may have the byte quantity as 2*8 then the word count is 8.This implies there are 8 register, where each register can store 2 bytes    
    If there is no word count return "Not Found"
    Register Type: The register type has information about whether the register is a "holding register" or "input register". To identify the register type we can use the following information.
    - If the register is only read register(R) or (RO), then it is an "input register". 
    - If the register is both Read and Write (R-W or R/W). Then it is a "holding register".
    - It can also be in the form of a Function code. If the function code is 0x04, then it is an "input register", all other function codes are "holding register".
    - It can be also obtained from the Address Type. Address type 10000-19999 range or Address type 1X is a "Discrete Input". If the address is in 40000-49999 range or Address type 4X then it is a "Holding register".If the address is in 30000-39999 range or  or Address type 3x then it is a "Input register".
    Function code - 0x04 or Read only register is a "input register"
    Function code - 0x03,0x06,0x16,0x10 or Read write register is a "holding register"
    Bit Information: This is information about what each bit of the register address represents. If any bit information is present in the text then extract that information in the bit_info key and value should be a string that contains the bit information. Keep the high bits and the low bits seperate, if they are seperated in the text. Extract only bit related information. Valid bit information follows this pattern: [number]- [description], [number]- [description], ..., default: [number].If no bit information is present then return "Not Found".
    Scale factor: The scale factor is a ratio that converts the process variable range to the scaled integer range. It can also be called scale or factor or ratio. The default scale factor is 1. 
    Unit: The unit the information is expressed in. Keep the unit just like it is present in the table. Use the same symbols and words. Do change the unit or expand. It unit not present return "Not Found".
    Component Name: The complete component name associated with the register address. Extract the complete and the most detailed component name only in English. The component should be as descriptive and if that means taking data from other columns to make a descriptive and complete component name. Sometimes the component name could be of a cluster, rack or pack, in that case ALWAYS prefix the cluster/rack/pack information along with the component name. 
    Make sure all the extracted information is in English for all the extracted keys.
    The register address should be in key called register_address. The data type should be in the key called data_type, word count should be in the key called word_count, the register type should be in the key register_type, the bit information in the bit_info key, the scale factor should be in scale and the component name should be in the key component_name. Report only the JSON file.
     If the text has no mobbus register address return an empty list. Don't come up with modbus register address or use the address within the example tags if no addresses are present in the given text.
    Remember the given text is a table. First row is the header. Use the header in the begining of the text to identify the contents of the table.
    Don't return any text other than the list of dictionaries.
    </guide>
    Here are some important rules for the extraction:component.
    - Do NOT convert address from Dec to Hex or Hex to Dec. No converting register address. This is a very important rule and can not be ignored.
    - Always extract address in Decimal if present. Do not use Hex address if Decimal address are present. Only if decimal register addresses are not in the text, then extract Hexadecimal.
    - Do not add decimal points to the addresses. Always present them as whole numbers without any decimal points.
    - Only if the headers and titles look relevant to the data of modbus register extract it. 
    - Don't use the data in the headers. The register address should be in the data, not in the header.
    - If you are unsure that there is a valid register address present in the given text return an empty list.
    - If any of the keys don't have value in the text, then return "Not Found" for that key
    - All extracted content MUST be only be in english, this is NOT optional. Non-English text should NOT be present in the the extracted text.
    - Think step by step.
    
    Think step by step in thinking tags. Look for decimal register address first, only in the absence of the decimal address use the Hex address. Put your response in <response></response> tags.
    Here is the text from which you need to extract the modbus register address. Use the information in title to get the register type for the register.
    <title>
    {title}
    </title>
    <customization>
    {customization}
    </customization>
    <text>
    {text_llm}
    </text>
    Assistant: <response>

    """
    logger.info("Sending batches of data to LLM")
    response = client.messages.create(
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model = model_id,
        
    )
    return response.content[0].text


def extract_text_between_braces(text):
    pattern = re.compile(r'\[\s*\{\s*(.*?)\s*\}\s*\]', re.DOTALL)  # Match '[', optional spaces and newlines, '{', optional spaces and newlines, everything in between, optional spaces and newlines, '}', and ']'
    match = pattern.search(text)
    if match:
        extracted_text = match.group(1)
        return extracted_text
    else:
        return None

    
def convert_llm_res_to_df(result_copy, titles_cleaned):
    """ Generates a dataframe from the responses"""
#     logger.info("Converting the json to dataframe")
    # Initialize an empty list to store DataFrames
    dfs_res = []
    i=0
    # Iterate over the list of texts
    for res in result_copy:
#         logger.info("*"*90)
        logger.info("Processing: %s", i)

        # Extract the JSON substring
        json_data = extract_text_between_braces(res)

        # Load JSON data into a list of dictionaries
        if json_data != None:
            json_data = '[{' + json_data + "}]"
            json_data = json_data.replace("#", '-')
    #         logger.info(json_data)
            lines = json_data.split('\n')
            lines = [line.split('#')[0].strip() for line in lines]
            json_data = '\n'.join(lines)

            try:
                # Create DataFrame
                df = pd.DataFrame(eval(json_data), columns=["register_address", "data_type","word_count", "register_type", "bit_info","scale_factor", "unit", "component_name"])
                df['captions'] = titles_cleaned[i]
                # Append the DataFrame to the list
                dfs_res.append(df)
            except Exception as e:
                logger.info(json_data)
                logger.info ("Error!!, can't process record %s", e)

        i = i + 1

    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(dfs_res, ignore_index=True)

    # Display the final DataFrame
#     display(final_df)
    return final_df

def clean_df_after_extraction(final_df_copy):
    """" Basic House keeping"""
#     logger.info("Clean up and organising")
    # Drop if register address, data type and bit info is na
    final_df_copy['register_address'] = final_df_copy['register_address'].replace({None: pd.NA})
    final_df_copy['register_address'] = final_df_copy['register_address'].dropna()
    final_df_copy  = final_df_copy.dropna(how= 'all', subset = ['component_name', 'data_type'])
    # Replace forward and backward slashes with hyphens
    final_df_copy['component_name'] = final_df_copy['component_name'].str.replace(r'/', '-', regex=True)
    final_df_copy['component_name'] = final_df_copy['component_name'].str.replace(r'\\', '-', regex=True)

    # for more uniformity
    final_df_copy['data_type'] = final_df_copy['data_type'].str.lower()

    # Rempve cells with no register address
    final_df_copy = final_df_copy[final_df_copy['register_address'] != '']

    # Drop duplicate - because sometime when it does not find the data in the text
    # It uses the register addresses used in the prompt
    # final_df_copy = final_df_copy.drop_duplicates(keep=False)
    # Make it neat and clean
    final_df_copy.reset_index(inplace = True, drop = True)
    final_df_copy = final_df_copy.dropna(subset = [ "data_type","register_type"], how='all')
    return final_df_copy

def ask_claude_to_expand(text_llm):
    prompt = f""" 
    Your task is to expand a range of register addresses provided in a text, which is actual a table where each cell is seperated by ',', and each row is seperated by '\n' into individual entries, each representing a specific version. 
    The table includes a range of register address can be a range of register (e.g., 0x07E0-0x07EF).
    Your goal is to generate a list where each entry corresponds to a unique entry of register address specified in the range. 
    All the expanded register will have the same data type and can be found in the data_type column. 
    For the word count, we need to pay specific attention. Each MODBUS data is 2 words(16 bits) in length. If the data type is U16 then the word count will be 1 for each register. There can be a case that the word count is given for entire range of registers, then to get the word count for indiviual register divide the given word count by number of registers. 
    So take a look at the data type and the word count. If the data type is U16 and the word count is 1, then all the expanded registers will also have the same value.
    Keep in mind the following relation between Data type and word count for a given register
    For data type INT16/I16/S16, word count is 1
    UINT16/U16, 16-bit unsigned integer (1 word)
    INT32/S32, 32-bit signed integer (2 words)
    UINT32/U32,32-bit unsigned integer (2 words)
    INT64/S64/I64, 64-bit signed integer (4 words)
    UINT64/U64, 64-bit unsigned integer (4 words)
    Float32/F32, 32-bit value (2 words)
    Try to think about what case of word count you are dealing with and how to expand it. 
    The component name for the expanded register should unique for each register. Try to make it by adding a suitable suffix.
    Here is an example
    <example>
    if the text is as given below. The first line is the header, and the second line is the register information.
    register_address,data_type,word_count,register_type,bit_info,scale_factor,unit,component_name,reg_id
    10104~10105,s32,2,holding register,None,0.1,kvar,Reactive power,reactive_power
    The response for this expansion will be as shown below - 
    10104,s32,2,holding register,None,0.1,kvar,Reactive power 1,reactive_power_1
    10105,s32,2,holding register,None,0.1,kvar,Reactive power 2,reactive_power_2
    </example>
    When expanding keep the format of the data. If the data type is in bytes. Then divided the byte by the num pof register that has been expanded for the word count.Ensure that each entry is accurately expanded from the register range, provide the new component name for each entry. Your output should provide a comprehensive breakdown of each the component that needs to be expanded, facilitating easy access and reference for further analysis or implementation. Return a only list of dictionaries within the <response> </response> tags and nothing else. The keys should be the column names or header.
    If the previous response already contains the last address in the range then return an empty list.
    If the data type is not found or empty or invalid then return an empty list.
    <text>
    {text_llm}
    </text>
    """
    
    response = client.messages.create(
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model = model_id,
    )
    return response.content[0].text

def ask_claude_to_expand_cont(text_llm, output_last_llm_call, num_of_register_to_expand):
    prompt = f""" 
    Your task is to expand a range of register addresses provided in a text, which is actual a table where each cell is seperated by ',', and each row is seperated by '\n' into individual entries, each representing a specific version. 
    The table includes a range of register address (e.g., 0x07E0-0x07EF).
    There is also the output from the previous LLM call. Continue the expanding from where it stopped for the last llm call and expand for the of number of registers specified. The registers in the output_last_llm_call should not be included in the new response. It MUST be continuation.
    Your goal is to generate a list where each entry corresponds to a unique entry of register address specified in the range. 
    For the word count, we need to pay specific attention. Each MODBUS data is 2 words(16 bits) in length. If the data type is U16 then the word count will be 1 for each register. There can be a case that the word count is given for entire range of registers, then to get the word count for indiviual register divide the given word count by number of registers. 
    So take a look at the data type and the word count. If the data type is U16 and the word count is 1, then all the expanded registers will also have the same value.
    Keep in mind the following relation between Data type and word count for a given register
    For data type INT16/I16/S16, word count is 1
    UINT16/U16, 16-bit unsigned integer (1 word)
    INT32/S32, 32-bit signed integer (2 words)
    UINT32/U32,32-bit unsigned integer (2 words)
    INT64/S64/I64, 64-bit signed integer (4 words)
    UINT64/U64, 64-bit unsigned integer (4 words)
    Float32/F32, 32-bit value (2 words)
    Try to think about what case of word count you are dealing with and how to expand it. 
    Don't use the word count given in the table. Look at the data type and use the information above and get the word count.
    The component name for the expanded register should unique for each register. Try to make it by adding a suitable suffix.
    Here is an example
    <example>
    if the text is as given below. The first line is the header, and the second line is the register information.
    register_address,data_type,word_count,register_type,bit_info,scale_factor,unit,component_name,reg_id
    10104~10105,s32,2,holding register,None,0.1,kvar,Reactive power,reactive_power
    The response for this expansion will be as shown below - 
    10104,s32,2,holding register,None,0.1,kvar,Reactive power 1,reactive_power_1
    10105,s32,2,holding register,None,0.1,kvar,Reactive power 2,reactive_power_2
    </example>
    Here is another example
        <example>
    if the text is as given below. The first line is the header, and the second line is the register information.
    register_address,data_type,word_count,register_type,bit_info,scale_factor,unit,component_name,reg_id
    20104~20106,s32,2,holding register,None,0.1,kvar,Cluster n temperature, cluster_n_temperature
    The response for this expansion will be as shown below - 
    20104,s32,2,holding register,None,0.1,kvar,Cluster 1 temperature,cluster_1_temperature
    20105,s32,2,holding register,None,0.1,kvar,Cluster 2 temperature,cluster_2_temperature
    20106,s32,2,holding register,None,0.1,kvar,Cluster 3 temperature,cluster_3_temperature
    so for the example above the correct way to expand is cluster_1_temperature and not cluster_1_temperature_1. The temperature is for each cluster. The are many cluster not temperatures.
    </example>
    The component name for the expanded register should unique for each register. Try to make it by adding a suitable suffix.
    You must NOT abruptly end the response, if the output has reached the max token size. Go to the last complete expanded register and give output in the specified format.
    If the previous response already contains the last address in the range then return an empty list.
    If the data type is not found or empty or invalid then return an empty list.
    Always return a complete list of dictionaries. It should not be cut or miss the brackets.Pay attention and think about how do this.
    <text>
    {text_llm}
    </text>
    <last_output>
    {output_last_llm_call}
    </last_output>
    <num_of_reg>
    {num_of_register_to_expand}
    </num_of_reg>
    """
    
    response = client.messages.create(
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model= model_id,

    )
    return response.content[0].text

def get_register_to_expand(df_extracted):
    df_extracted['register_address'] = df_extracted['register_address'].astype('str')
    # Identify all the registers that have '-' or '~', usually these are registers need to expanded
    df_need_to_exp = df_extracted[df_extracted['register_address'].str.contains('-|~|Regs|to|,| ', na=False)]

    # Don't expand rows where the component name is reserve
    df_need_to_exp = df_need_to_exp[~df_need_to_exp['component_name'].str.contains('reserve|Reserve|Total Reservation|total reservation', case=False, na=False)]
    
     # Don't expand rows where the component name is Not found
    df_need_to_exp = df_need_to_exp[~df_need_to_exp['component_name'].str.contains('Not Found|not found', case=False, na=False)]
    
         # Don't expand rows where the register address is Not found
    df_need_to_exp = df_need_to_exp[~df_need_to_exp['register_address'].str.contains('Not Found|not found', case=False, na=False)]
    
    # logger.info or use the filtered rows
    logger.info(df_need_to_exp.shape)
#     display(df_need_to_exp)

    # Convert 'word count' to numeric, invalid parsing will be set as NaN
    df_need_to_exp['word_count'] = pd.to_numeric(df_need_to_exp['word_count'], errors='coerce')

    # Drop rows with NaN values in 'word count'
    df_need_to_exp = df_need_to_exp.dropna(subset=['word_count'])
    df_need_to_exp = df_need_to_exp.reset_index(drop = True)
    # df_need_to_exp = df_need_to_exp.replace("Not Found", None)
    # Replace empty strings with NaN for uniformity
    df_need_to_exp = df_need_to_exp.replace('', np.nan)

    # Define the conditions for rows to be dropped
    conditions = (df_need_to_exp['word_count'].isin(['Not Found', 'not found']) | df_need_to_exp['word_count'].isna()) | \
                 (df_need_to_exp['data_type'].isin(['Not Found', 'not found']) | df_need_to_exp['data_type'].isna())

    # Drop the rows based on the conditions
    df_need_to_exp_cleaned = df_need_to_exp[~conditions]
    logger.info("Shape of the dataframe that needs to be extracted: %s",df_need_to_exp_cleaned.shape)
    return df_need_to_exp_cleaned
    

def drop_rows(df_extracted):
    # Condition for register_address containing any of '-|~|Regs|to|,| '
    address_condition = df_extracted['register_address'].str.contains('-|~|Regs|to|,| ')
    reserved_condition = df_extracted['component_name'].str.contains('reserve|Reserve|Total Reservation|total reservation')
    combined_condition = address_condition & ~reserved_condition
    result_df = df_extracted[~combined_condition]
    return result_df
    
def get_expanded_registers(df_need_to_exp_cleaned):
    """ Makes the LLM call to expand the register ranges"""
    results_expand = []
    # Sort Based on Word count, as we cannot expand the register based on high word count. 
    try:
        df_need_to_exp_cleaned = df_need_to_exp_cleaned[df_need_to_exp_cleaned['word_count'] < 592]
        # Send to the LLM
        for index,row in df_need_to_exp_cleaned.iterrows():
            header_string = ','.join([f"{key}" for key, value in row.items()])# Get the column header
            row_string = ','.join([f"{value}" for key, value in row.items()])# Get the values in the column
            final_str_expand = header_string+'\n'+row_string
            logger.info("Expanding the registers: %s", final_str_expand)
            wc = row['word_count']
            logger.info("Starting word count: %s", wc)
            time.sleep(2)
            res = ""
            if wc > 8 and wc < 592:
                num_of_reg = 8
                while(wc > 0):
                    res_expand = ask_claude_to_expand_cont(final_str_expand, res, num_of_reg) 
                    logger.info(res_expand)
                    res = res_expand
                    results_expand.append(res_expand)
                    wc = wc-num_of_reg
                    logger.info("Word Count : %s", wc)
            else:
                res_expand = ask_claude_to_expand(final_str_expand)
                results_expand.append(res_expand)
    except Exception as e:
        logger.info("Error: ", e)
    return results_expand

# Post process the response from the dataframe. Extract the list of dictionaries, convert to a dataframe. 
# Concat all the dataframes that were generated because of expansion of the register.
# Finally concat this to the original dataframe
# Drop all the rows that need to be expanded

def process_exp_reg_output(results_expand):
    # Initialize an empty list to store DataFrames
    dfs_exp = []
    # Counter
    i = 0
#     filtered_df = final_df_copy.copy()
    if len(results_expand) > 0:
        # Iterate over the list of texts
        for res_ in results_expand:
            logger.info("Processing record: %s", i)

            # Extract the JSON substring
            json_data_exp = extract_text_between_braces(res_)

            # Load JSON data into a list of dictionaries
            if json_data_exp != None:
                json_data_exp = '[{' + json_data_exp + "}]"
                json_data_exp = json_data_exp.replace('null', '""')

                try:
                    # Create DataFrame
                    df_exp = pd.DataFrame(ast.literal_eval(json_data_exp), columns=["register_address", "data_type","word_count", "register_type", "bit_info","scale_factor", "unit", "component_name"])
                    dfs_exp.append(df_exp)
                except Exception as e:
                    logger.info(json_data_exp)
                    logger.info ("Error!!, can't process record", e)
            i=i+1

        # Concatenate all DataFrames into a single DataFrame
        df_exp_fin = pd.concat(dfs_exp, ignore_index=True)
        return df_exp_fin

# Usually when the table with title is extracted with textract after unmerging the title cell, then all the cell of the table contains the 
# the same content which is the title. So in that case use that
from collections import Counter
def are_all_columns_same(row):
    most_common = Counter(row).most_common(1)
    return len(row) > len(set(row)), most_common

# Header should not have repetitive cells
# Should not contain numbers.
def is_header(row):
    # Check for unique values
    if len(set(row)) != len(row):
        return False
    # Check for numbers and empty cells
    for cell in row:
        if pd.isnull(cell) and not str(cell).isalpha():
            return False
    return True

def identify_header(df):
    for idx, row in df.iterrows():
        if is_header(row):
            return idx
    return None

# Function to clean JSON string
def clean_json_string(json_string):
    # Remove tabs and other unwanted special characters
    cleaned_string = re.sub(r'[\t\n\r]', ' ', json_string)
    # Optionally, remove other special characters or replace multiple spaces with a single space
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
    return cleaned_string
    

def ask_claude_bit_information(text_llm, title):
    prompt = f""" 
    Your task is to identify the component for which the given text contains Bit information, You also need to extract the bit information(like Bit 0, Bit 1, etc) and what it represents. 
    The given text is actually a table where the cells are seperated by '\t' and each row is seperated by '\n'. The first row may or may not be the header. If it is the header, use the header to identify the correct column to be extracted. The text is a complete table for one component. All the bit information in the text is assocaiated with one component. 
    Use the text may have the component name and the bit information for that component. Use these guideline to find the component name.
    <guide>
    - The text in the title could be the component name, but a more relevant component name could be in the column. Pick the more correct one. Sometimes it can be NotFound or NoTitle. In that case the component_name can be in one of the columns.
    - Sometimes the component name in the tag could have additional words, like Appendix or Reference. In that case return the component name without these words.
    - Here can't be multiple components in a given text. 
    - Don't return multiple component names for a text. 
    - The component name will never start with "bit/BIT". Do not pick the name that start with bit or Bit or BIT. If component name is not found return Not Found. 
    - Component name can only be in English. Don't return component name in any other language
    - Component name can be with parenthesis/brackets, in that case return only the name without the parenthesis.
    - Don't use colon (:) while summarizing the bit info. 
    - Extract the complete component name from the text. Remove any special characters, but keep the case and numbers intact
    </guide>
    Always return a list of dictionaries where the dictionary with in the list will contain two keys, the component_name and the bit_info. The key component_name will contain the name of the component, and the key bit_info will only have the information in the table summarized, which what each bit represents. Don't add the component name to bit_info. You must return the value of bit_info as a string and NOT as a dictionary.
    Extract the entire component name, including the numbers exactly as in the text. Everthing must be in English. Don't extract text in any other language. Don't add any additional text. Just whatever is in the input text.
    The list of dictionary must contain these two keys. If there is no value then return "Not Found". Return your response in <response></response> tags. 
    <Title>
    {title}
    </Title>
    <text>
    {text_llm}
    </text>
    """
    
    response = client.messages.create(
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model = model_id,

    )
    return response.content[0].text
    
    
def process_bit_info(csv_list_bit):
    i=0
    nrows = 0
    responses = []
    text_header = ""
    df_bit_fin = pd.DataFrame()
    dfs_bit = []
    count = 0
    comp_name = ""
    if len(csv_list_bit) > 0:
        for csv in csv_list_bit:
            logger.info("=====================++++++++++++++++++++============================")
            logger.info("Processing csv : %s", csv)
            try:
                # Read the csv file into a dataframe
                df_bit = pd.read_csv(csv , skiprows = nrows, sep = ',', dtype=str, index_col=False, quoting=2, on_bad_lines='skip', keep_default_na=False)
                comp_name = df_bit['Title'].iloc[0].split(".")[-1]
                df_bit = df_bit.drop('Title', axis=1)
                logger.info("comp_name: %s", comp_name)

            except Exception as e:
                logger.info("ERROR!!!!!:Could not process csv file ",  e)

            #Get the header for the table in the sheet. 
            text_header = ""

            # Check why this is done
            for column in df_bit.columns:
                df_bit[column] = df_bit[column].astype(str) + '\t'
            df_bit['\n'] = '\n' 

            rows_per_split = 32
            #If the dataframe is too long then break it up -
            if (len(df_bit) > 32):
                # Calculate the total number of splits needed
                total_splits = len(df_bit) // rows_per_split + (1 if len(df_bit) % rows_per_split != 0 else 0)
                for k in range(total_splits):
                    logger.info("Splitting dataframe")
                    start_index = k * rows_per_split
                    end_index = min((k + 1) * rows_per_split, len(df_bit))
                    split_df = df_bit.iloc[start_index:end_index].copy()
                    dfBitAsString = split_df.to_string(header=False, index=False)
                    dfBitAsString = text_header + dfBitAsString
                    res = re.sub(' +', ' ', dfBitAsString)
                    res  = re.sub('\"', '', res)
                    res  = re.sub(':', '', res)
                    response = ask_claude_bit_information(res, comp_name)
                    logger.info(response)
                    responses.append(response)
                    dfs_bit.append(df_bit)
            else:
                logger.info("Small dataframe")
                dfBitAsString = df_bit.to_string(header=False, index=False)
                dfBitAsString = text_header + dfBitAsString
                res = re.sub(' +', ' ', dfBitAsString)
                res  = re.sub('\"', '', res)
                res  = re.sub(':', '-', res)
                logger.info(res)
                response = ask_claude_bit_information(res, comp_name)
                logger.info(response)
                responses.append(response)
                dfs_bit.append(df_bit)
        # Concatenate all DataFrames into a single DataFrame
        df_bit_fin = pd.concat(dfs_bit, ignore_index=True)
        # Display the final DataFrame
        return responses

# Function to clean JSON string
def clean_json_string(json_string):
    # Remove tabs and other unwanted special characters
    cleaned_string = re.sub(r'[\t\n\r]', ' ', json_string)
    # Optionally, remove other special characters or replace multiple spaces with a single space
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
    return cleaned_string

def process_llm_bit_response(responses):
#     df_bit_fin = pd.DataFrame()
    dfs_bit = []
    i=0
    words_to_replace = ["appendix","Appendix","reference","Reference"]
    pattern = '|'.join(words_to_replace)
    # Regular expression to match repeated words
    pattern_repeat = r'\b(\w+)\1\b'

    if len(responses) > 0:
        for response in responses:
            # Extract the JSON substring
            json_data_bit = extract_text_between_braces(response)
            # Load JSON data into a list of dictionaries
            if json_data_bit != None:
                logger.info("Processing record: %s", i)
                json_data_bit = '[{' + json_data_bit + '}]'
                json_data_bit = json_data_bit.replace("'", "\"")
                json_data_bit_cleaned = clean_json_string(json_data_bit)
                try:
                    # Create DataFrame
                    records = json.loads(json_data_bit_cleaned)
                    df_bit = pd.DataFrame(records, columns=["component_name", "bit_info"])
                    dfs_bit.append(df_bit)
                except Exception as e:
                    logger.info(json_data_bit)
                    logger.info ("Error!!, can't process record", e)
            i=i+1

        # Concatenate all DataFrames into a single DataFrame
        df_bit_fin = pd.concat(dfs_bit, ignore_index=True)
        # Display the final DataFrame
        return df_bit_fin
    
def update_bit_info_with_partial_match(df1, df2):
    updated_rows = []
    for idx, row in df1.iterrows():
        logger.info("Row: %s", row)
        component_name = row['component_name']
        logger.info("Component name: %s", component_name)
        # Check for partial matches in df2
        match = df2['component_name'].str.contains(component_name, case=False, na=False)
        
        if match.any():
            matching_indices = df2.index[match].tolist()
            # Update all matches in df2
            for matching_index in matching_indices:
                df2.at[matching_index, 'bit_info'] = row['bit_info']
            updated_rows.append(idx)

    # Remove updated rows from df1
    df1 = df1.drop(updated_rows).reset_index(drop=True)
    return df1, df2



# Function to remove the '0x' prefix
def remove_hex_prefix(address):
    if str(address).startswith('0x'):
        return str(address).replace('0x', "")
    elif str(address).startswith('0X'):
        return str(address).replace('0X', "")
    return address


# Function to standardize the data type
def standardize_data_type(data_type):
    return type_mapping.get(data_type, data_type)

# Function to extract the numeric part from the filename for sorting
def extract_number(file_path):
    filename = file_path.split('/')[-1]  # Get the filename from the path
    number = int(filename.split('_')[-1].split('.')[0])  # Extract the number
    return number