# bit_utils.py

import pandas as pd
import re
import os

def bit_has_at_least_two_non_empty_cells(row):
    non_empty_cells = [cell for cell in row if cell is not None and str(cell).strip().lower() != 'nan']
    return len(non_empty_cells) >= 2

def cell_contains_entirely_numbers(text):
    if isinstance(text, (int, float)):
        return True
    if isinstance(text, str) and text.isdigit():
        return True
    return False

def bit_contains_some_english(text):
    def is_english(char):
        try:
            char.encode('ascii')
            return True
        except UnicodeEncodeError:
            return False
            
    if isinstance(text, str):
        english_chars = sum(1 for char in text if is_english(char))
        total_chars = len(text)
        return english_chars / total_chars > 0.7
    return True

def has_mostly_unique_values(row):
    values = [str(cell).strip().lower() for cell in row if cell is not None and str(cell).lower() != 'nan']
    unique_values = set(values)
    return sum(values.count(value) for value in unique_values) - len(unique_values) < 1

def is_valid_bit_header_row(row):
    if not bit_has_at_least_two_non_empty_cells(row):
        return False
    
    for cell in row:
        if cell_contains_entirely_numbers(cell):
            return False
    
    return any(bit_contains_some_english(cell) for cell in row if cell is not None) and has_mostly_unique_values(row)

def identify_bit_header_row(df):
    for row_index, row in enumerate(df.itertuples(index=False), start=1):
        if is_valid_bit_header_row(row):
            latest_header_row = row_index - 1
#             print(f"Identified header row at index: {latest_header_row + 1}")
            return latest_header_row
    return None

def identify_bit_tableidentifier(df, header_row_index):
    if header_row_index > 0:
        table_identifier = df.iat[header_row_index - 1, 0]
        if pd.isna(table_identifier) or (isinstance(table_identifier, pd.Series) and table_identifier.isna().all()):
            table_identifier = "UnidentifiedComponent"
    else:
        table_identifier = "UnidentifiedComponent"
    
    #print(table_identifier)
    #table_identifier_clean = ''.join(char for char in str(table_identifier) if ord(char) < 128)
    table_identifier_clean = re.sub(r'[[\\/*?:"<>|]]：', "", table_identifier)
    table_identifier_clean = table_identifier_clean[:100]
    return table_identifier, table_identifier_clean

# def drop_columns_more_than_90_empty(df):
#     threshold = 0.9
#     num_rows = len(df)
#     columns_to_drop = [col for col in df.columns if df[col].isna().sum() / num_rows > threshold]
#     df = df.drop(columns=columns_to_drop)
#     return df

# def drop_columns_more_than_90_empty(df):
#     threshold = 0.9
#     num_rows = len(df)
#     #print(f"Number of rows: {num_rows}")
#     columns_to_drop = []

#     for col in df.columns:
#         num_missing = df[col].isna().sum()
#         #print(f"Column: {col}, Missing: {num_missing}, Threshold: {threshold * num_rows}")
#         if num_rows!=0:
#             if num_missing / num_rows > threshold:
#                 columns_to_drop.append(col)
        
#     print(f"Columns to drop (more than 90% empty): {columns_to_drop}")
#     df = df.drop(columns=columns_to_drop)
#     return df

def contains_non_english_numerical(text):
    valid_characters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/[]-_(){}!@#$%^&*+=|\\;:'\",.<>?~` ")
    if isinstance(text, str):
        non_english_numerical_chars = sum(1 for char in text if char not in valid_characters)
        total_chars = len(text)
        return non_english_numerical_chars / total_chars > 0.5 if total_chars > 0 else False
    return False

# def should_drop_based_on_non_english_content(df, column_name):
#     non_english_content_count = df[column_name].apply(contains_non_english_numerical).sum()
#     total_elements = len(df[column_name])
#     return non_english_content_count / total_elements > 0.5 if total_elements > 0 else False

def should_drop_based_on_non_english_content(df, column_name):
    non_english_content_count = df[column_name].apply(lambda x: contains_non_english_numerical(x) if pd.notna(x) else False).sum()
    total_elements = df[column_name].notna().sum()  # Count only non-NaN values
    return non_english_content_count / total_elements > 0.5 if total_elements > 0 else False

def drop_columns_more_than_90_empty(df):
    threshold = 0.9
    num_rows = len(df)
    columns_to_drop = []

    for col in df.columns:
        num_missing = df[col].isna().sum()
#         print(f"Column: {col}, Missing: {num_missing}, Total rows: {num_rows}, Threshold: {threshold * num_rows}")
        if num_rows != 0:
            if num_missing > (threshold*num_rows):
                columns_to_drop.append(col)
    
    #print(f"Columns to drop (more than 90% empty): {columns_to_drop}")
    df = df.drop(columns=columns_to_drop)
    return df

def table_cleanup(df):
    df.columns = df.columns.astype(str)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#     print("Step 1: Drop unnamed columns")
    #print(df.head())
    
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
#     print("Step 2: Drop all-NaN columns and rows")
    #print(df.head())
    
    df = drop_columns_more_than_90_empty(df)  # Drop columns more than 90% empty
#     print("Step 3: Drop columns more than 90% empty")
    #print(df.head())
    
    columns_to_drop = [col for col in df.columns if should_drop_based_on_non_english_content(df, col)]
#     print("Columns to drop based on non-English content: ", columns_to_drop)
    
    df = df.drop(columns=columns_to_drop)
#     print("Step 4: Drop columns based on non-English content")
    #print(df.head())
    
    return df

# def table_cleanup(df):
#     df.columns = df.columns.astype(str)
#     df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
#     print("Done1")
#     df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
#     print("Done2")
#     df = drop_columns_more_than_90_empty(df)  # Drop columns more than 90% empty
#     print("Done3")
#     columns_to_drop = [col for col in df.columns if should_drop_based_on_non_english_content(df, col)]
#     print("Columns to drop: ", columns_to_drop)
#     df = df.drop(columns=columns_to_drop)
#     return df

# def table_cleanup(df):
#     # Backup column headers to reassign after cleanup
#     column_headers = df.columns
    
#     # Drop unnamed columns
#     df = df.loc[:, [col for col in df.columns if col is not None and not col.startswith('Unnamed')]]
    
#     # Drop rows and columns that are all NaN
#     df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
    
#     # Drop columns more than 90% empty
#     df = drop_columns_more_than_90_empty(df)
    
#     # Identify columns to drop based on non-English content
#     columns_to_drop = [col for col in df.columns if col is not None and should_drop_based_on_non_english_content(df, col)]
#     print("Columns to drop: ", columns_to_drop)
#     df = df.drop(columns=columns_to_drop)

#     # Reassign the column headers after cleanup
#     df.columns = column_headers[:len(df.columns)]
    
#     return df

def is_numerical_row(row):
    for cell in row:
        if cell is None or cell == '':
            continue
        try:
            float(cell)
        except ValueError:
            return False
    return True

def remove_numerical_top_rows(df):
    while not df.empty and (is_numerical_row(df.iloc[0]) or df.iloc[0].apply(cell_contains_entirely_numbers).all()):
        df = df.iloc[1:]
    return df

def clean_and_save_tables(tables, sheet_name, base_dir_path):
    bit_dir = os.path.join(base_dir_path, "Bit")
    found_bit_table = False
    table_identifier = "Not found"
    table_identifier_clean =  "Not found"
    bit_table_number = 1
    for i, table in enumerate(tables):
        if not table:
            continue
        
        try:
            df = pd.DataFrame(table)
            #print("Intial Df: ",df)
            header_row_index = identify_bit_header_row(df)
            #print("Header row Index: ", header_row_index)
            if header_row_index is not None:
                # Determine table identifier
                df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')
                table_identifier, table_identifier_clean = identify_bit_tableidentifier(df, header_row_index)
                print("!!!"*50)
                print("Table Identifier: ", table_identifier)
                print("table_identifier_clean: ", table_identifier_clean)
                print("!!!"*50)
                # Set the identified header row as the columns
                df.columns = df.iloc[header_row_index]
                df = df.iloc[header_row_index + 1:]
                #print("Intermediate Df: ", df)
                #print("header and data set")
            df = table_cleanup(df)
            #print("table cleanup done")
            df = remove_numerical_top_rows(df)
            #print("Removing done")
            if df.empty or df.shape[0] == 0:
                continue
            df = df.drop_duplicates()
            if df.empty or df.shape[0] <= 1:
                continue
                
            if table_identifier_clean == "UnidentifiedComponent":
                csv_file_name = f"{sheet_name}_{table_identifier_clean}-{bit_table_number}.csv"
            else:
                if ":" in table_identifier:
                    c_name = table_identifier.split(":")[-1]
                    df['Component Name'] = c_name
                    csv_file_name = f"{sheet_name}_{c_name}.csv"
                else:
                    df['Component Name'] = table_identifier_clean
                    csv_file_name = f"{sheet_name}_{table_identifier_clean}.csv"
            
            csv_path = os.path.join(bit_dir, csv_file_name)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)
            found_bit_table = True
            bit_table_number += 1

        except Exception as e:
            print(f"Error processing table in sheet '{sheet_name}': {e}")

    if not found_bit_table:
        print(f"No bit tables found in sheet: {sheet_name}")



# def separate_tables(ws, last_row_first_table):
#     tables = []
#     current_table = []
#     inside_first_table = True
#     found_table = False

#     for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
#         if i <= last_row_first_table:
#             continue  # Skip rows until the end of the first table
        
#         if ws.row_dimensions[i].hidden:
#             continue

#         if any(cell is not None for cell in row):
#             if inside_first_table:
#                 inside_first_table = False
            
#             # Check for start of a new table
#             if len([cell for cell in row if cell is not None]) == 1 or any(row.count(cell) > 1 for cell in row if cell is not None):
#                 if current_table:
#                     tables.append(current_table)
#                     current_table = []
#                 current_table.append(row)
#                 found_table = True
#             else:
#                 current_table.append(row)
#                 found_table = True
#         else:
#             if current_table:
#                 tables.append(current_table)
#                 current_table = []
    
#     if current_table:
#         tables.append(current_table)
    
#     if not found_table:
#         print("No bit tables found")

#     return tables


# def separate_tables(ws, last_row_first_table):
#     tables = []
#     current_table = []
#     inside_first_table = True
#     found_table = False
#     empty_row_count = 0

#     for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
#         if i <= last_row_first_table:
#             continue  # Skip rows until the end of the first table

#         if ws.row_dimensions[i].hidden:
#             continue

#         if any(cell is not None for cell in row):
#             empty_row_count = 0
#             if inside_first_table:
#                 inside_first_table = False
# #             # Check for start of a new table

#             non_empty_cells = [cell for cell in row if cell is not None and str(cell).strip().lower() != 'nan']
#             if len(non_empty_cells) <= 2 or all(cell is None for cell in row) or any(row.count(cell) > 2 for cell in row if cell is not None):
#                 if current_table:
#                     tables.append(current_table)
#                     current_table = []
#                 current_table.append(row)
#                 found_table = True
#             else:
#                 current_table.append(row)
#                 found_table = True
#         else:  # Entirely empty row
#             if current_table:
#                 tables.append(current_table)
#                 current_table = []
#                 found_table = True  # Mark that we found at least one table

#     if current_table:
#         tables.append(current_table)

#     if not found_table:
#         print("No bit tables found")

#     return tables


def drop_repeated_columns(df):
    """
    Drop repeated columns that have exactly the same values.
    Keep only the first instance of each repeated column.
    """
    df_t = df.T
    df_t = df_t.drop_duplicates()
    return df_t.T

def is_single_value_row(row_values):
    non_na_values = [cell for cell in row_values if cell is not None and str(cell).strip().lower() != 'nan']
    return len(set(non_na_values)) == 1

def separate_tables(ws, last_row_first_table):
    tables = []
    current_table = []
    inside_first_table = True
    found_table = False

    for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if i <= last_row_first_table:
            continue  # Skip rows until the end of the first table

        if ws.row_dimensions[i].hidden:
            continue

        row_values = [cell for cell in row]

        # Drop repeated columns
        row_df = pd.DataFrame([row_values])
        row_df = drop_repeated_columns(row_df)
        row_values = row_df.values.tolist()[0]

        if any(cell is not None for cell in row_values):
            if inside_first_table:
                inside_first_table = False
            
            non_empty_cells = [cell for cell in row_values if cell is not None and str(cell).strip().lower() != 'nan']
            
            # Check for breakage conditions
            if len(non_empty_cells) <= 2 or all(cell is None for cell in row_values) or is_single_value_row(row_values):
                if current_table:
                    tables.append(current_table)
                    current_table = []
                current_table.append(row_values)
                found_table = True
            else:
                current_table.append(row_values)
                found_table = True
        else:  # Entirely empty row
            if current_table:
                tables.append(current_table)
                current_table = []
                found_table = True  # Mark that we found at least one table
        
    if current_table:
        tables.append(current_table)

    if not found_table:
        print("No bit tables found")

    return tables