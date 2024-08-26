# src/excel_llm_processing_utils.py


import re
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from excel_extraction.Utils.app_logger import logger
from anthropic import AnthropicBedrock
client = AnthropicBedrock()

# model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Dictionary to map abbreviations to full type names
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

def process_csv(csv_list_):
    """ Read's the output of the Excel extraction that are stored in csv's and build a list of dataframes """
    logger.info("Processing CSV...")
    dfs = []
    i=0
    nrows = 0
    for csv in csv_list_:
#         print("=====================++++++++++++++++++++============================")
        logger.info("=====================++++++++++++++++++++============================")
#         print("Processing csv :", csv)
        logger.info("Processing csv : %s", csv)
        
        i=i+1
        try:
            df = pd.read_csv(csv , skiprows = nrows, sep = ',', dtype=str, index_col=False, quoting=2, on_bad_lines='skip', header= 'infer', keep_default_na=False)

        except Exception as e:
#             print("ERROR!!!!!:Could not process csv file ",   e)
            logger.info("ERROR!!!!!:Could not process csv file %s",  e)

        nan_value = float("NaN")
        # Replace '\t' and '\r\n' with empty spaces in column headers
        df.columns = [col.replace('\t', '').replace('\r\n', '') for col in df.columns]
        df.iloc[:,-1:] = df.iloc[:,-1:].replace("", nan_value)
        df = df.replace("", nan_value)
        df = df.dropna(how='all').dropna(how='all', axis=1)
        df = df.replace(nan_value,"")
        df = df.drop_duplicates()
        df['Sheet Name'] = Path(csv).stem
        dfs.append(df)
    return dfs

def chunk_clean_preLLM(dfs):
    """ Prepare the data for the LLM. Set's the header and create batches"""
    logger.info("Preparing data for LLM")
    headers = [] # holds the headers
    prev_header = None
    tables = []
    text_header = ""
    count = 0
    title = ""
    titles_cleaned = []
    for i in range(0, len(dfs)):
#         print("\n")
#         print("*"*50)
#         print("Processing i: ", i)
#         print("Lenght of the dataframe: ", len(dfs[i]))
        
        logger.info("\n")
        logger.info("*"*50)
        logger.info("Processing i: %s", i)
        logger.info("Lenght of the dataframe: %s", len(dfs[i]))

        temp_df = dfs[i]
        logger.info("Columns: %s", temp_df.columns)
        if "Text Above Header" in temp_df.columns:
            title = temp_df["Text Above Header"].iloc[0]
            temp_df = temp_df.drop("Text Above Header", axis = 1)

        #Get the header for the table in the sheet. 
        text_header = ""
        for j in range(0,len(dfs[i].columns)):
            text_header =  text_header + str(dfs[i].columns[j]) + '\t'
        text_header = text_header + '\n'

        # Check why this is done
        for column in temp_df.columns:
            temp_df[column] = temp_df[column].astype(str) + '\t'
        temp_df['\n'] = '\n' 

        # Split if the table is too big
    #     if len(dfs[i]) > 10:
        if len(temp_df) > 20:

            logger.info("+++++++Splitting++++++++++++++")
            # Number of rows for each split
            rows_per_split = 20

            # Calculate the total number of splits needed
            total_splits = len(temp_df) // rows_per_split + (1 if len(temp_df) % rows_per_split != 0 else 0)
            logger.info("Total splits: %s",total_splits)
            for k in range(total_splits):
                logger.info("Splitting dataframe")
                start_index = k * rows_per_split
                end_index = min((k + 1) * rows_per_split, len(temp_df))
                split_df = temp_df.iloc[start_index:end_index].copy()
                dfAsString = split_df.to_string(header=False, index=False)
                dfAsString = text_header + dfAsString
                res = re.sub(' +', ' ', dfAsString)
                res  = re.sub('\"', '', res)
                res  = re.sub(':', '', res)
                titles_cleaned.append(title)
                tables.append(res)
                count = count + 1
        else:
            logger.info("Small dataframe")
            dfAsString = temp_df.to_string(header=False, index=False)
            dfAsString = text_header + dfAsString
            res = re.sub(' +', ' ', dfAsString)
            res  = re.sub('\"', '', res)
            res  = re.sub(':', '-', res)
            tables.append(res)
            titles_cleaned.append(title)

            count = count + 1
    
    logger.info("Count: %s",count)
    return tables, titles_cleaned


# First LLM call to the main register extraction
def ask_claude(text_llm, title, customization):
    prompt = f"""You will be acting as an expert electronic engineer for the  company Flexgen, who is going to extract the modbus register address and other details associated with register like register type, data type that the register holds and other details that are given below.
    Your goal is to extract the following from the given text.
    - Modbus register address
    - Data type of the register
    - Word count
    - Register type
    - Bit information of what each register bit depicts
    - Scale factor
    - Unit of the Component 
    - Sheet Name
    - Component name of the component whose register address is extracted
  
    You will return a list of dictionaries with the keys of the dictionary being the register address, data type of the registers, word count, the register type, the bit info of what each register bit depicts, scale factor, unit, sheet name and the component name of the component whose register address is extracted.
    You should stick to extracting only these information from the text. There is also a title to the table. It could contain information of the register type(input register or holding register). Use the customization as an guideline to generate the final output. Here is the guidance document you should use as a reference to identifying and extracting the register address: 
    <guide>
    The given text is a table. The cells are seperated by '\t' and '\n' represents the end of a given row. The first row is the header of the table. Extract the register address, data type, word count, bit information, scale factor, unit, sheet name and component name and return a list of dictionaries with seven keys. The keys are register_address, data_type, register_type, bit_info, scale_factor, unit and component_name. Carefully understand the criteria for the extraction of Modbus register and the associated information.
    1. Register addresses: Look at the header to identify the register address. The register address can be in Decimal or Hexadecimal. Hexadecimal address usually have '0x' in their address. If there are 2 address columns, it will be identified as Dec or Hex. If the text contains addresses in decimal(dec/DEC), then always extract the decimal address unless it is specified in the customization. If there are NO decimal addresses, but there are hexadecimal(hex/HEX) address. then extract the hexadecimal address. Register address will not be made up of english words. If there is no identifiable register address, return None. Register address cannot be called BIT or have english words. They can have characters, if they are hexadecimal addresses. There can also be range of valid address, Identify and extract register ranges, which may be presented in two ways:
      a. Explicitly stated ranges (e.g., "0x0001-0x0015 will store software version")
      b. Consecutive rows that represent a range (e.g., "0x0001 Software version 1" followed by "0x0015 Software version 15")
    For consecutive rows, only combine them into a range if they meet ALL of the following criteria:
      a. They appear to be part of the same logical group (e.g., software versions or cluster information)
      b. The range covers a significant number of registers (more than 5)
      c. They are not already listed as individual entries in the text
    Do NOT combine registers into ranges in the following cases:
      a. When two consecutive registers address have the same description then do NOT make it a range. 
      b. When consecutive registers have different descriptions
      c. When it's clear that each register represents a distinct piece of information, even if they are related
    For ranges, include the start and end register addresses and a description of what the range represents.
    Note: consecutive rows need not be consecutive register address. If the address range between consecutive rows is greater than 2 and the component name is the same, then make it a range otherwise keep them as in the text. 
    If there's any ambiguity do not combine registers into a range, keep them as separate entries.

    2. Data Type: This is the type of data that is stored in the register. It can be either directly specified in the text as unsigned or singed int(U16 or I16 or U32) or the bytes representation can be given. Use the hints in the customization to correctly extract the value. If the data type is given Byte Quantity, then check the number of register address it is applicable for and then calculate the byte quantity for each register.
    Some of the commonly present data types are INT16, UINT16, ENUM, BOOL, BITMAP. There could be more. 
    Sometime the data type can be a bit. This is true for Discrete Input register. If the bit information is limited only to bit 0 and bit 1, then the data type could be bit.
    There can be cases where there are two register are there for one component and the data type is U32. In that case make an entry for  each register as u16 and make the wordcount 1 for each register.  This can be extended to all data types, like s32, i32 etc. If there is no data type then return "Not Found".

    3. Word Count: 1 word = 16 bits = 2 bytes.The word count field represents the number of 16-bit registers requested to be read or written. 
    For a given register, the word count can be calculated if the data type is given or byte quantity.
    - INT16/I16/S16, 16-bit signed integer or 2 bytes, so word count will be 1
    - UINT16/U16/, 16-bit unsigned integer or 2 bytes, so word count will be 1
    - INT32/S32/I32, 32-bit signed integer or 4 bytes, so word count will be 2
    - UINT32/U32,32-bit unsigned integer or 4 bytes, so word count will be 2
    - INT64/S64/I64, 64-bit signed integer or 8 bytes, so word count will be 4
    - UINT64/U64, 64-bit unsigned integer or 8 bytes, so word count will be 4
    - Float32/F32, 32-bit float value or 4 bytes, so word count will be 2
    - If the data type is U16*8 then the word count will be 8 
    - In some case you may have the byte quantity as 2*8 then the word count is 8.This implies there are 8 register, where each register can store 2 bytes  
    - If the address is a range, that has the first address and the last address then the word count will number of register in the range multiplied by the word count for each register.
    - If there is no word count return "Not Found".

    4. Register Type: The register type has information about whether the register is a "holding register" or "input register". To identify the register type we can use the following information.
    - If the register is only read register(R) or (RO), then it is an "input register". 
    - If the register is both Read and Write (R-W or R/W). Then it is a "holding register".
    - It can also be in the form of a Function code. If the function code is 0x04, then it is an "input register", all other function codes are "holding register".
     - It can be also obtained from the Address Type. Address type 10000-19999 range or Address type 1X is a "Discrete Input". If the address is in 40000-49999 range or Address type 4X then it is a "holding register".If the address is in 30000-39999 range or  or Address type 3x then it is a "input register". 
    Function code - 0x04 or Read only register is a "input register".
    Function code - 0x03,0x06,0x16,0x10 or Read write register is a "holding register".
    - If there is no register type then return "Not Found".
    
    5. Bit Information: This is information about what each bit of the register address represents. If any bit information is present in the text then extract that information in the bit_info key and value should be a string that contains the bit information. Keep the high bits and the low bits seperate, if they are seperated in the text. Extract only bit related information. Valid bit information follows this pattern:[number]- [description], [number]- [description], ..., default: [number].If no bit information is present then return "Not Found".
    
    6. Scale factor: The scale factor is a ratio that converts the process variable range to the scaled integer range. It can also be called scale or factor or ratio. The default scale factor is 1. 
    
    7. Unit: The unit the information is expressed in. Keep the unit just like it is present in the table. Use the same symbols and words. Do change the unit or expand. It unit not present return "Not Found".
    
    8. Sheet Name: It is the name of the excel sheet where the text was extracted from.
    
    9. Component Name: The complete component name associated with the register address. Extract the complete and the most detailed component name only in English. The component name should be descriptive and this could mean taking data from other columns to make as descriptive and complete as possible. In some cases the component name could be of a cluster, rack or pack, in that case ALWAYS prefix the cluster/rack/pack information along with the component name. 
    Make sure all the extracted information is in English for all the extracted keys.
    The register address should be in key called register_address. The data type should be in the key called data_type, word count should be in the key called word_count, the register type should be in the key register_type, the bit information in the bit_info key, the scale factor should be in scale, sheet name should be in a key called sheet_name and the component name should be in the key component_name. Report only the JSON file.
    If the text has no modbus register address return an empty list. Don't come up with modbus register address or use the address within the example tags if no addresses are present in the given text.
    Remember the given text is a table. First row is the header. Use the header in the begining of the text to identify the contents of the table.
    Don't return any text other than the list of dictionaries.
    </guide>
    Here are some important rules for the extraction:
    - Do NOT convert address from Dec to Hex or Hex to Dec. No converting register address. This is a very important rule and can not be ignored.
    - if not specified in the customization, on what type of registers to extract then always extract address in Decimal if present. Do not use Hex address if Decimal address are present. Only if decimal register addresses are not in the text, then extract Hexadecimal, only if there are no specific instruction in the customization. 
    - Do not add decimal points to the addresses. Always present them as whole numbers without any decimal points.
    - Only if the headers and titles look relevant to the data of modbus register extract it. 
    - Don't use the data in the headers. The register address should be in the data, not in the header.
    - Don't make address range only with two unique register addresses. Leave them as present in the text.
    - If any of the keys don't have value in the text, then return "Not Found" for that key
    - All extracted content MUST be only be in english, this is NOT optional. Non-English text should NOT be present in the the extracted text.
    - Think step by step.
    
    Think step by step in thinking  <thinking></thinking>tags. Look for decimal register address first, only in the absence of the decimal address use the Hex address. Put your response in <response></response> tags.
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
        model=model_id,
        
    )
    return response.content[0].text


def extract_text_between_braces(text):
    """ Extracts the list of dictionaries in LLM response"""
    logger.info("Extracting LLM response")
    pattern = re.compile(r'\[\s*\{\s*(.*?)\s*\}\s*\]', re.DOTALL)  # Match '[', optional spaces and newlines, '{', optional spaces and newlines, everything in between, optional spaces and newlines, '}', and ']'
    match = pattern.search(text)
    if match:
        extracted_text = match.group(1)
        return extracted_text
    else:
        return None
    
def convert_llm_res_to_df(results):
    """ Generates a dataframe from the responses"""
    logger.info("Converting the json to dataframe")
    # Initialize an empty list to store DataFrames
    dfs = []
    i=0
    # Iterate over the list of texts
    for res in results:
        logger.info("*"*90)
        logger.info("Processing: %s", i)
        
        # Extract the JSON substring
        json_data = extract_text_between_braces(res)
        
        # Load JSON data into a list of dictionaries
        if json_data != None:
            json_data =  "[{" + json_data + "}]"
            json_data = json_data.replace("#", '-')
            json_data = json_data.replace("\\n", ';')
            json_data = json_data.replace('null', '""')

            try:
                # Create DataFrame
                df = pd.DataFrame(eval(json_data), columns=["register_address", "data_type","word_count", "register_type", "bit_info","scale_factor", "unit", "sheet_name", "component_name"])
                # Append the DataFrame to the list
                dfs.append(df)
            except Exception as e:
                logger.info("Data that could not be extracted %s", json_data)
                logger.info ("Error!!, can't process record", e)

        i = i + 1
    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(dfs, ignore_index=True)

    # Display the final DataFrame
    return final_df

def clean_df_after_extraction(final_df):
    """" Basic House keeping"""
    logger.info("Clean up and organising")
    # Convert regiter add to str
    final_df['register_address'] = final_df['register_address'].astype('str')
    # for more uniformity
    final_df['data_type'] = final_df['data_type'].str.lower()
    # generate the reg_id, there use
    final_df['component_name'] = final_df['component_name'].str.replace(":", " ")
    final_df['component_name'] = final_df['component_name'].str.replace("/", " ")
    final_df['component_name'] = final_df['component_name'].str.replace("\\", " ")
    final_df['data_type'] = final_df['data_type'].replace("0", np.NaN)
    final_df['register_type'] = final_df['register_type'].replace("0", np.NaN)
    final_df['register_address'] = final_df['register_address'].ffill()
    # Remove cells with no register address
    final_df = final_df[final_df['register_address'] != '']
    # Make it neat and clean
    final_df.reset_index(inplace = True, drop = True)
    return final_df

# Once all the register extracted from the vendor doc, look for register ranges and expand those. Register range expansion are limited to certain word count. Also all rows where the data type and word count are zero are dropped.
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
    When expanding keep the format of the data. If the data type is in bytes. Then divided the byte by the num pof register that has been expanded for the word count.Ensure that each entry is accurately expanded from the register range, provide the new component name for each entry. Your output should provide a comprehensive breakdown of each the component that needs to be expanded, facilitating easy access and reference for further analysis or implementation. Return a only list of dictionaries within the <response> </response> tags and nothing else. The keys should be the column names or header.
    You must NOT abruptly end the response, if the output has reached the max token size. Go to the last complete expanded register and give output in the specified format.
    Always return a complete list of dictionaries. It should not be cut or miss the brackets.Pay attention and think about how do this.
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
#         model="anthropic.claude-3-sonnet-20240229-v1:0",
        model=model_id,
    )
    return response.content[0].text

# Particulary to expanding huge ranges. Prev response is passed , and LLM is asked to continue till the range is met.
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
    The component name for the expanded register should unique for each register. Try to make it by adding a suitable suffix.
    You must NOT abruptly end the response, if the output has reached the max token size. Go to the last complete expanded register and give output in the specified format.
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
        model=model_id,
    )
    return response.content[0].text

def ask_claude_to_get_starting_register(text_llm):
    prompt = f""" 
    Your task is to get the starting address in  range of register addresses provided in a text. The text is actually a table where each cell is seperated by ',', and each row is seperated by '\n' into individual entries.
    The table includes a range of register address (e.g., 0x07E0-0x07EF).
    Your goal is to generate a list where the entry corresponds to a starting entry of register address specified in the range. 
    Here is an example
    <example>
    if the text is as given below. The first line is the header, and the second line is the register information.
    register_address,data_type,word_count,register_type,bit_info,scale_factor,unit,component_name,reg_id
    10104~10105,s32,2,holding register,None,0.1,kvar,Reactive power,reactive_power
    The response for this expansion will be as shown below - 
    10104,s32,2,holding register,None,0.1,kvar,Reactive power,reactive_power
    </example>
    Always return a complete list of dictionaries.
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
        model=model_id,
    )
    return response.content[0].text

def get_register_to_expand(df_extracted):
    """ Get the register that need to be expanded"""
    df_need_to_exp = pd.DataFrame()
    # Identify all the registers that have '-' or '~', usually these are registers need to expanded
    df_need_to_exp = df_extracted[df_extracted['register_address'].str.contains('-|~|Regs|to|,| ', na=False)]
    
    # Don't expand rows where the component name is reserve
    df_need_to_exp = df_need_to_exp[~df_need_to_exp['component_name'].str.contains('reserve|Reserve|Total Reservation|total reservation', case=False, na=False)]

    # Print or use the filtered rows
    logger.info("Shape of the dataframe that needs to expanded: %s", df_need_to_exp.shape)

    # Convert 'word count' to numeric, invalid parsing will be set as NaN
    df_need_to_exp['word_count'] = pd.to_numeric(df_need_to_exp['word_count'], errors='coerce')

    # Drop rows with NaN values in 'word count'
    df_need_to_exp = df_need_to_exp.dropna(subset=['word_count'])
    df_need_to_exp = df_need_to_exp.reset_index(drop = True)
    df_need_to_exp = df_need_to_exp.replace("", "Not Found")
    
    # Define the conditions for rows to be dropped
    conditions = (df_need_to_exp['word_count'].isin(['Not Found', 'not found']) | df_need_to_exp['word_count'].isna()) | \
                 (df_need_to_exp['data_type'].isin(['Not Found', 'not found']) | df_need_to_exp['data_type'].isna())

    # Drop the rows based on the conditions
    df_need_to_exp_cleaned = df_need_to_exp[~conditions]
    return df_need_to_exp_cleaned

def drop_rows(df_extracted):
    # Condition for register_address containing any of '-|~|Regs|to|,| '
    address_condition = df_extracted['register_address'].str.contains('-|~|Regs|to|,| ')
    reserved_condition = df_extracted['component_name'].str.contains('reserve|Reserve|Total Reservation|total reservation')
    combined_condition = address_condition & ~reserved_condition
    result_df = df_extracted[~combined_condition]
    return result_df

def get_expanded_registers(df_exp):
    """ Makes the LLM call to expand the register ranges"""
    results_expand = []
    # Sort Based on Word count, as we cannot expand the register based on high word count. 
    try:
        df_need_to_exp = df_exp[df_exp['word_count'] < 513]
        # Send to the LLM
        for index,row in df_need_to_exp.iterrows():
            header_string = ','.join([f"{key}" for key, value in row.items()])# Get the column header
            row_string = ','.join([f"{value}" for key, value in row.items()])# Get the values in the column
            final_str_expand = header_string+'\n'+row_string
            logger.info("Expanding the registers: %s", final_str_expand)
            wc = row['word_count']
            res = ""
            if wc > 16 and wc < 513:
                num_of_reg = 16
                while(wc > 0):
                    logger.info("Word count more than 16")
                    logger.info("Number of register: %s", num_of_reg)
                    logger.info("Word Count: %s", wc)
                    res_expand = ask_claude_to_expand_cont(final_str_expand, res, num_of_reg) 
                    res = res_expand
                    results_expand.append(res_expand)
                    wc = wc-num_of_reg
            elif wc > 2 and wc <= 16 :
                logger.info("Word count less than 16")
                res_expand = ask_claude_to_expand(final_str_expand)
                results_expand.append(res_expand)
            elif wc == 2 and '32' in row['data_type'] :
                logger.info("Word count is equal to 2")
                res_expand = ask_claude_to_get_starting_register(final_str_expand)
                results_expand.append(res_expand)
            elif wc == 2 and '32' not in row['data_type']:
                logger.info("Word count is equal to 2, and data type is not 32")
                res_expand = ask_claude_to_expand(final_str_expand)
                results_expand.append(res_expand)
    except Exception as e:
        logger.info("Error: %s", e)
    return results_expand

# LLM call to consolidate the bit info
def ask_claude_bit_information(text_llm, comp_name):
    prompt = f""" 
    Your task is to identify the component for which the given text contains Bit information, You also need to extract the bit information(like Bit 0, Bit 1, etc) and what it represents. 
    The given text is actually a table where the cells are seperated by '\t' and each row is seperated by '\n'. The text is a complete table for one component. All the bit information in the text is assocaiated with one component. 
    The text will have the component name and the bit information for that component. Use these guideline to find the component name.
    <guide>
    - The text in the component name tag could be the component name. Sometimes it can be NotFound or NoTitle. In that case the component_name can be in one of the columns.
    - Sometimes the component name in the tag could have additional words, like Appendix or Reference. In that case return the component name without these words.
    - Here can't be multiple components in a given text. 
    - Don't return multiple component names for a text. 
    - Present the information exactly as in the text. Do NOT form sentence that are not in the text.
    - The component name will never start with "bit/BIT". Do not pick the name that start with bit or Bit or BIT. If component name is not found return Not Found. 
    - Component name can only be in English. Don't return component name in any other language
    - Component name can be with parenthesis/brackets, in that case return only the name without the parenthesis.
    - Extract the complete component name from the text. Remove any special characters, but keep the case and numbers intact
    </guide>
    Always return a list of dictionaries where the dictionary with in the list will contain two keys, the component_name and the bit_info. The key component_name will contain the name of the component, the key bit_info will have the information in the table summarized, which what each bit represents and you must return the value of bit_info as a string and NOT as a dictionary. and  key sheet_name will contain the name of the sheet. 
    Extract the entire component name, including the numbers exactly as in the text. Everthing must be in English. Don't extract text in any other language. Don't add any additional text. Just whatever is in the input text.
    The list of dictionary must contain these two keys. If there is no value then return "Not Found". Return your response in <response></response> tags. 
    <comp_name>
    {comp_name}
    </comp_name>
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
        model=model_id,
    )
    return response.content[0].text


def process_bit_info(csv_list_bit):
    """Process the bit info and makes the LLM call"""
    i=0
    nrows = 0
    responses = []
    text_header = ""
    count = 0
    for csv in csv_list_bit:
        logger.info("=====================++++++++++++++++++++============================")
        logger.info("Processing csv :%s", csv)
        entire_comp_name = Path(csv).stem
        comp_name = entire_comp_name.split("_")[-1]
        sheet_name = entire_comp_name.split("_")[0]
        
        if "UnidentifiedComponent" in comp_name:
            comp_name = "Component name could be in the table, as a column."
        try:
            # Read the csv file into a dataframe
            df_bit = pd.read_csv(csv , skiprows = nrows, sep = ',', dtype=str, index_col=False, quoting=2, on_bad_lines='skip', header= 'infer', keep_default_na=False)
            df_bit['sheet_name'] = sheet_name
        except Exception as e:
            logger.info("ERROR!!!!!:Could not process csv file: %s ",  e)

        #Get the header for the table in the sheet. 
        text_header = ""
        for j in range(0,len(df_bit.columns)):
            text_header =  text_header + str(df_bit.columns[j]) + '\t'
        text_header = text_header + '\n'
        
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
                print(response)
                responses.append(response)
        else:
            logger.info("Small dataframe")
            dfBitAsString = df_bit.to_string(header=False, index=False)
            dfBitAsString = text_header + dfBitAsString
            res = re.sub(' +', ' ', dfBitAsString)
            res  = re.sub('\"', '', res)
            res  = re.sub(':', '-', res)
            response = ask_claude_bit_information(res, comp_name)
            responses.append(response)
    return responses

def update_bit_info_with_partial_match(df1, df2):
    """The bits summarized have to be matched with the component name in the main dataframe and plugged in bit info cell."""
    updated_rows = []
    for idx, row in df1.iterrows():
        sheet_name = row['sheet_name']
        component_name = row['component_name']
        logger.info("sheet_name :%s", sheet_name)
        logger.info("component_name :%s", component_name)
        
        # Check for partial matches in df2
        match = df2['component_name'].str.contains(component_name, case=False, na=False) & df2['sheet_name'].str.contains(sheet_name, case=False, na=False)

#         if component_name in df2['component_name'].tolist() and sheet_name in df2['sheet_name'].tolist():
#             logger.info("Match found")
            
        if match.all() :
            logger.info("Match found")
            matching_indices = df2.index[match].tolist()
            # Update all matches in df2
            for matching_index in matching_indices:
                df2.at[matching_index, 'bit_info'] = row['bit_info']
            updated_rows.append(idx)

    # Remove updated rows from df1
    df1 = df1.drop(updated_rows).reset_index(drop=True)
    return df1, df2

def process_llm_bit_response(responses):
    """Convert the LLM response to dataframe"""
    df_bit_fin = pd.DataFrame()
    dfs_bit = []
    i=0

    if len(responses) > 0:
        for response in responses:
            # Extract the JSON substring
            json_data_bit = extract_text_between_braces(response)
            # Load JSON data into a list of dictionaries
            if json_data_bit != None:
                logger.info("Processing record: %s", i)
                json_data_bit = '[{' + json_data_bit + '}]'
                json_data_bit = json_data_bit.replace("'", "\"")
                json_data_bit = json_data_bit.replace("\t",  " ")
                json_data_bit = json_data_bit.replace("\n", " ")

                try:
                    # Create DataFrame
                    df_bit = pd.DataFrame(ast.literal_eval(json_data_bit), columns=["component_name", "bit_info", "sheet_name"])
                    dfs_bit.append(df_bit)
                except Exception as e:
                    logger.info(json_data_bit)
                    logger.info ("Error!!, can't process record: %s", e)

            i=i+1

        # Concatenate all DataFrames into a single DataFrame
        df_bit_fin = pd.concat(dfs_bit, ignore_index=True)
        return df_bit_fin
    
def ask_claude_to_expand_bit_information(text_llm):
    prompt = f""" 
    I have a row from a dataframe containing Modbus register information. The first row is the header of the table. Each cell in the row is separated by a tab character ('\t'). When the row contains bit information, I need you to expand it into multiple rows, one for each bit. Please follow these rules:

    1. If the row contains bit information, create a main row with the original information, then create additional rows for each bit.
    2. For bit rows, use these fields same as the main row. The fields that remain the same are 'register_address','register_type', 'scale_factor', 'unit', 'sheet_name' and 'reg_id'
    3. For the new rows set the data type for bit rows to 'bit'
    4. Under bit information put the bit number in the original data.
    5. Use the bit description as the component name for the new rows.
    
    Please process the following row and return the result as a list of dictionaries. Each dictionary should have the same keys are the original row.
    Always return a complete list of dictionaries. It should not be cut or miss the brackets.Pay attention and think about how do this.
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
        model=model_id,
    )
    return response.content[0].text

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
