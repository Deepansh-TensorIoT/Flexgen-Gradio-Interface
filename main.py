import gradio as gr
import pandas as pd
import openpyxl
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import logging
import os
#from pdf_extraction.extract import get_interim_data, extract_data_with_llm
from excel_extraction.EE_LLM_driver import llm_driver_function
from excel_extraction.EE_main_driver import run_extraction
from Database_functions import (fetch_models_from_dynamodb, 
                                fetch_vendors_from_dynamodb,
                                fetch_versions_from_dynamodb, 
                                add_vendor_to_dynamodb, 
                                add_processed_vendors_to_ddb,
                                fetch_comparision_models,
                                fetch_comparision_vendors,
                                fetch_comparision_versions,
                                add_pdf_vendor_specific_inputs_to_ddb,
                                add_excel_vendor_specific_inputs_to_ddb,
                                add_models_to_dynamodb,
                                add_versions_to_dynamodb)


from pdf_processing_new.PDF_extraction_main_driver import get_interim_data, extract_data_with_llm
from S3_utils import read_latest_csv_from_s3


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_sheet_names(file):
    """ Get sheet names from the excel file"""
    # Load the Excel file with openpyxl
    wb = openpyxl.load_workbook(file.name, read_only=True)
    # Get the visible sheet names
    visible_sheet_names = [sheet.title for sheet in wb.worksheets if sheet.sheet_state == 'visible']
    return ", ".join(visible_sheet_names)

def clear_comparision_tab_llm():
    return {
        #compare_comparision_stats_df: gr.update(value = None),
        #compare_rows_added: gr.update(value = None),
        #compare_rows_removed: gr.update(value = None),
        compare_comparision_stats: gr.update(visible = False),
        compare_view_comparision: gr.update(visible = False)
    }

def clear_comparision_tab():
    return {
        #comparision_stats_df: gr.update(value = None),
        #rows_added: gr.update(value = None),
        #rows_removed: gr.update(value = None),
        comparision_stats: gr.update(visible = False),
        view_comparision: gr.update(visible=False),
    }


def clear_inputs_pdf():
    return {
        pdf_vendor_input: gr.update(value="Clou"),
        pdf_model_input: gr.update(value="Model 1"),
        pdf_version_input: gr.update(value="V1"),
        pdf_file_input: gr.update(value=None),
        start_page: gr.update(value=None),
        start_page_table: gr.update(value=None),
        end_page: gr.update(value=None),
        appendix_page: gr.update(value=None),
        appendix_start_table: gr.update(value=None),
        customization: gr.update(value=""),
        interim_output_group_pdf: gr.update(visible=False),
        run_llm_conversion_btn_group_pdf: gr.update(visible=False),
        final_output_group_pdf: gr.update(visible=False),
        download_btn_group_pdf: gr.update(visible=False)
    }

def clear_inputs_excel():
    return {
        excel_dropdown_inputs[0]: gr.update(value="Clou"),
        excel_dropdown_inputs[1]: gr.update(value="Model 1"),
        excel_dropdown_inputs[2]: gr.update(value="V1"),
        excel_file_input: gr.update(value=None),
        lastrowlist: gr.update(value=""),
        interim_output_excel_text: gr.update(value=""),
        sheet_names_text: gr.update(value=""),
        sheet_names_ui: gr.update(visible=False),
        interim_output_group_excel: gr.update(visible=False),
        run_llm_conversion_btn_group_excel: gr.update(visible=False),
        final_output_group_excel: gr.update(visible=False),
        download_btn_group_excel: gr.update(visible=False)
    }

def append_suffix_to_filename(file_path, suffix, extension=None):
    file_name, file_extension = os.path.splitext(file_path)
    new_file_name = file_name.split('/')[-1] + suffix + (file_extension if extension is None else extension)
    return new_file_name

def toggle_ui(file_type):
    if file_type == "pdf":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

def handle_custom_vendor(value):
    if value not in vendors:
        vendors.append(value)
        add_vendor_to_dynamodb(value)
        return [gr.update(choices=vendors, value=value)]*2
    return [gr.update(choices=vendors, value=value)]*2

def handle_custom_model(vendor_name, value):
    if value not in models:
        models.append(value)
        add_models_to_dynamodb(vendor_name, value)
        return [gr.update(choices=models, value=value)]*2
    return [gr.update(choices=models, value=value)]*2

def handle_custom_version(vendor_name, model_name, value):
    if value not in versions:
        versions.append(value)
        add_versions_to_dynamodb(vendor_name, model_name, value)
        return [gr.update(choices=versions, value=value)]*2
    return [gr.update(choices=versions, value=value)]*2

def make_group_visible():
    return gr.update(visible=True)

def get_interim_output(vendor_input, file_input, start_page, start_page_table, end_page, appendix_page, appendix_start_table, customization, pdf_model_input,pdf_version_input):
    return get_interim_data(vendor_name=vendor_input, input_file_path=file_input.name, start_page=start_page, start_page_table = start_page_table, end_page=end_page, appendix_start_page=appendix_page, app_table=appendix_start_table, customization=customization, model_name = pdf_model_input, version_name = pdf_version_input)

def get_final_output(vendor_input, file_input, customization, pdf_model_input,pdf_version_input):
    return extract_data_with_llm(vendor_name=vendor_input,
                                input_file_path=file_input.name,
                                customization=customization,
                                model_name = pdf_model_input,
                                version_name = pdf_version_input
                                )

def export_csv(file_input, df):
    download_file_name = append_suffix_to_filename(file_input.name, "_extracted", ".xlsx")
    #df.to_excel(download_file_name, index=None)
    with pd.ExcelWriter(download_file_name, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        text_format = workbook.add_format({'num_format': '@'})
        worksheet.set_column('A:A', None, text_format)
    return gr.File(value=download_file_name, visible=True)

def handle_add_vendor(new_vendor):
    if new_vendor not in vendors:
        updated_vendors = add_vendor_to_dynamodb(new_vendor)
        return [gr.update(choices=updated_vendors, value=new_vendor)] * 2
    return [gr.update(choices=vendors, value=new_vendor)] * 2

def handle_comparision_dropdowns(vendor,model,version, df):
    df_size = df.shape[0]
    comparision_vendors, comparision_models, comparision_versions = add_processed_vendors_to_ddb(vendor,model,version, df_size)
    comparision_vendors = list(set(comparision_vendors))
    comparision_models = list(set(comparision_models))
    comparision_versions = list(set(comparision_versions))
    return [gr.update(choices = comparision_vendors), gr.update(choices = comparision_models), gr.update(choices = comparision_versions)]*2

def get_interim_output_excel(vendor_name, vendor_path, last_rows, model_input,version_input):
    return run_extraction(vendor_name, vendor_path.name, last_rows, model_input,version_input)

def get_final_output_excel(BASE_DIR, vendor_name, model_input,version_input, customization):
    return llm_driver_function(BASE_DIR, vendor_name, model_input,version_input, customization)

def run_file_comparision(vendor_name, first_model, first_version, vendor_name2, second_model, second_version):
    new = read_latest_csv_from_s3(vendor_name, first_model, first_version)
    old = read_latest_csv_from_s3(vendor_name2, second_model, second_version)
    # print("File 1:", new)
    # print("File 1:", old)
    
    
    new.drop(columns = ['reg_id'], inplace = True)
    old.drop(columns = ['reg_id'], inplace = True)
    
    # Replace special characters in column 'B' with spaces
    new['component_name'] = new['component_name'].str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    new['component_name'] = new['component_name'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    new['bit_info'] = new['bit_info'].str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    new['bit_info'] = new['bit_info'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Replace special characters in column 'B' with spaces
    old['component_name'] = old['component_name'].str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    old['component_name'] = old['component_name'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    old['bit_info'] = old['bit_info'].str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    old['bit_info'] = old['bit_info'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    merged_df = pd.merge(new, old, on=list(new.columns),
                        how="outer", 
                        indicator=True)
    
    
    # added entries are those that exist in new but not in old
    added_entries = merged_df[merged_df['_merge'] == 'left_only']

    # deleted entries are those that exist in old but not in new
    deleted_entries = merged_df[merged_df['_merge'] == 'right_only']
    
    total_old_rows = len(old)
    added_count = len(added_entries)
    deleted_count = len(deleted_entries)

    percent_change = ((added_count - deleted_count) / total_old_rows) * 100

    comaprison_df = pd.DataFrame({
        '% change': [percent_change],
        'rows added': [added_count],
        'rows deleted': [deleted_count]
    })
    
    return added_entries, deleted_entries, comaprison_df


def read_and_process_file(file_path):
    column_names = ["Name", "Address", "Register Type", "Data Type", "Bit", "Words", "Scale", "Unit", "Reg ID"]
    new_columns = ['component_name', 'register_address', 'register_type', 'data_type', 'bit_info', 'word_count', 'scale_factor', 'unit', 'reg_id']
    # Read the Excel file into a Pandas DataFrame
    try:
        df = pd.read_excel(file_path, header=None)
    except Exception as e:
        df = pd.read_csv(file_path, header=None)

    # Find the row index where the column names match
    for row_idx in range(min(20, len(df))):
        if all(col_name in list(df.iloc[row_idx]) for col_name in column_names):
            print("columns found!!")
            break
    new_df = df[row_idx:].reset_index(drop=True)  # Reset the index
    
    new_df.columns = new_df.iloc[0]  # Set the column names
    new_df.drop(columns=[col for col in new_df.columns if col not in column_names], axis=1, inplace=True)
    new_df = new_df.iloc[1:].dropna(axis=0, how='all')
    print(new_df)
    new_df = new_df.rename(columns=dict(zip(column_names, new_columns)))
    
    return new_df



def compare_run_file_comparision(vendor_name, first_model, first_version, compare_file_input):
    user_input_file_path = compare_file_input.name
    # logger.info("run_file_comparison()")
    new = read_latest_csv_from_s3(vendor_name, first_model, first_version)
    old = read_and_process_file(user_input_file_path)
    new['register_address'] = new['register_address'].astype(str)  # Convert to string
    old['register_address'] = old['register_address'].astype(str)
    merged_df = pd.merge(new, old, on=list(new.columns),
                         how="outer", 
                         indicator=True)
    
    new.drop(columns = ['reg_id'], inplace = True)
    new.drop(columns = ['reg_id'], inplace = True)

    # Replace special characters in column 'B' with spaces
    df['B'] = df['B'].str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    df['B'] = df['B'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # added entries are those that exist in new but not in old
    added_entries = merged_df[merged_df['_merge'] == 'left_only']

    # deleted entries are those that exist in old but not in new
    deleted_entries = merged_df[merged_df['_merge'] == 'right_only']
    
    total_old_rows = len(old)
    added_count = len(added_entries)
    deleted_count = len(deleted_entries)

    percent_change = ((added_count - deleted_count) / total_old_rows) * 100

    comaprison_df = pd.DataFrame({
        '% change': [percent_change],
        'rows added': [added_count],
        'rows deleted': [deleted_count]
    })
    
    return added_entries, deleted_entries, comaprison_df

# Fetch initial vendors from DynamoDB
vendors = fetch_vendors_from_dynamodb()
models = fetch_models_from_dynamodb()
model_versions = fetch_versions_from_dynamodb()
models = list(model_versions.keys())
# Collect unique versions
unique_versions = set()
for model, versions in model_versions.items():
    unique_versions.update(versions)
    print(f"Model: {model}, Versions: {versions}")

# Convert the set of unique versions to a list
versions = list(unique_versions)

#Values for dropdowns on comparision tab
comparision_vendors = list(set(fetch_comparision_vendors()))
comparision_models = list(set(fetch_comparision_models()))
comparision_versions = list(set(fetch_comparision_versions()))



with gr.Blocks(title="FlexGen - Auto Registers Extraction") as interface:
    with gr.Tabs():
        with gr.TabItem("Document Upload"):
            with gr.Row():
                file_type = gr.Radio(["pdf", "excel"], label="File Type", value="pdf")
                # new_vendor_input = gr.Textbox(label="New Vendor", placeholder="Enter new vendor name")
                # add_vendor_btn = gr.Button(value="Add Vendor")

            with gr.Group(visible=True) as pdf_ui:
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""<h2>File Details</h2>""")
                        pdf_vendor_input = gr.Dropdown(
                            choices=vendors,
                            label="Vendor",
                            value="Clou",
                            allow_custom_value=True,
                        )
                        pdf_model_input = gr.Dropdown(
                            choices=models,
                            label="Model",
                            value="Model 1",
                            allow_custom_value=True
                        )
                        pdf_version_input = gr.Dropdown(
                            choices=versions,
                            label="Version",
                            value="V1",
                            allow_custom_value=True
                        )
                        pdf_file_input = gr.File(label="Upload PDF file")
                    with gr.Column():
                        gr.HTML("""<h2>User Inputs</h2>""")
                        start_page = gr.Number(label="Start Page Number", value=None)
                        start_page_table = gr.Number(label="Start Page Table Number", value=None)
                        end_page = gr.Number(label="End Page Number", value=None)
                        appendix_page = gr.Number(label="Appendix Page Number", value=None)
                        appendix_start_table = gr.Number(label="Appendix Start Table Number", value=None)
                        customization = gr.Textbox(label="Vendor specific customization",
                                                value=None,
                                                placeholder="Example: Address type 3x are input registers", lines=10)
                with gr.Row():
                    clear_button_pdf = gr.Button(value="Clear", variant="secondary")
                    submit_btn_pdf = gr.Button(value="Submit", variant="primary")

                with gr.Group(visible=False) as interim_output_group_pdf:
                    with gr.Accordion("View Preprocessed Output"):
                        #interim_output_pdf = gr.Dataframe(label="Preprocessed Output")
                        interim_output_pdf_text = gr.Text(value="PDF file preprocessed successfully")

                with gr.Group(visible=False) as run_llm_conversion_btn_group_pdf:
                    run_llm_conversion_btn_pdf = gr.Button(value="Run LLM Conversion", variant="primary", scale=0)

                with gr.Group(visible=False) as final_output_group_pdf:
                    with gr.Accordion("View Final Output"):
                        final_output_pdf = gr.Dataframe(label="Final Output")
                
                with gr.Group(visible=False) as download_btn_group_pdf:
                    download_btn_pdf = gr.Button(value="Download Converted File", variant="primary", scale=0)
                    csv_pdf = gr.File(interactive=False, visible=False)

            with gr.Group(visible=False) as excel_ui:
                with gr.Row():
                    with gr.Column():
                        gr.HTML("""<h2>File Details</h2>""")
                        excel_dropdown_inputs = [
                            gr.Dropdown(
                            choices=vendors,
                            label="Vendor",
                            value="Clou",
                            allow_custom_value=True,
                        ),
                        gr.Dropdown(
                            choices=models,
                            label="Model",
                            value="Model 1",
                            allow_custom_value=True
                        ),
                        gr.Dropdown(
                            choices=versions,
                            label="Version",
                            value="V1",
                            allow_custom_value=True
                        )
                        ]
                        excel_file_input = gr.File(label="Upload Excel file")
                    with gr.Column():
                        gr.HTML("""<h2>Sheet Names and Last Row Input</h2>""")
                        get_sheet_names_button = gr.Button(value="Get Sheet Names", variant="primary", scale=0)
                        with gr.Group(visible=False) as sheet_names_ui:
                            sheet_names_text = gr.Textbox(label="Sheets in Excel File")

                        lastrowlist = gr.Textbox(label="Vendor specific list, Please enter No if you want to skip a sheet and 0 if no register table",
                                                placeholder="Example: [NO,NO,138,0,343]")
                        excel_customization = gr.Textbox(label="Vendor specific customization",
                                                value="All the registers type are holding registers",
                                                placeholder="Example: Address type 3x are input registers", lines=10)
                with gr.Row():
                    clear_button_excel = gr.Button(value="Clear", variant="secondary")
                    submit_btn_excel = gr.Button(value="Submit", variant="primary")

                with gr.Group(visible=False) as interim_output_group_excel:
                    with gr.Accordion("View Preprocessed Output"):
                        interim_output_excel_text = gr.Text(value="Excel file preprocessed successfully")

                with gr.Group(visible=False) as run_llm_conversion_btn_group_excel:
                    run_llm_conversion_btn_excel = gr.Button(value="Run LLM Conversion", variant="primary", scale=0)

                with gr.Group(visible=False) as final_output_group_excel:
                    with gr.Accordion("View Final Output"):
                        final_output_excel = gr.Dataframe(label="Final Output")
                        # excel_csv_path_final = gr.Text(value="Excel file preprocessed successfully")
                with gr.Group(visible=False) as download_btn_group_excel:
                    download_btn_excel = gr.Button(value="Download Converted File", variant="primary", scale=0)
                    csv_excel = gr.File(interactive=False, visible=False)
        with gr.TabItem("Compare LLM generated files"):
            with gr.Row():
                with gr.Row():
                    first_file = gr.Dropdown(
                                choices=comparision_vendors,
                                label="First File",
                                value="Clou",
                            )
                    first_model = gr.Dropdown(
                                choices=comparision_models,
                                label="First Model",
                                value="Model 1",
                            )
                    first_version = gr.Dropdown(
                                choices=comparision_versions,
                                label="First Version",
                                value="V1",
                    )
                    
            with gr.Row():
                with gr.Row():
                    second_file = gr.Dropdown(
                                choices=comparision_vendors,
                                label="Second File",
                                value="Clou",
                    )
                    second_model = gr.Dropdown(
                                choices=comparision_models,
                                label="Second Model",
                                value="Model 1",
                    )
                    second_version = gr.Dropdown(
                                choices=comparision_versions,
                                label="Second Version",
                                value="V1",
                    )
            with gr.Row():
                run_comparision_btn = gr.Button(value="Run Comparison", variant="primary", scale=-5)
                compare_clear_button = gr.Button(value = "Clear", variant = "secondary", scale = -5)
            
            with gr.Group(visible = False) as comparision_stats:
                comparision_stats_df = gr.Dataframe(label="Comparison Statistics")
                view_comp_btn = gr.Button(value="View Comparison", variant="primary", scale=-5)

            with gr.Group(visible = False) as view_comparision:
                rows_added = gr.Dataframe(label="Rows Added")
                rows_removed = gr.Dataframe(label="Rows Removed")

        with gr.TabItem("Compare LLM generated files vs User Generated files", visible = False):
            with gr.Row():
                with gr.Row():
                    compare_first_file = gr.Dropdown(
                                choices=comparision_vendors,
                                label="First File",
                                value="Clou",
                            )
                    compare_first_model = gr.Dropdown(
                                choices=comparision_models,
                                label="First Model",
                                value="Model 1",
                            )
                    compare_first_version = gr.Dropdown(
                                choices=comparision_versions,
                                label="First Version",
                                value="V1",
                    )
                    
            with gr.Row():
                with gr.Row():
                    compare_file_input = gr.File(label="Upload User Generated file")

            with gr.Row():
                compare_run_comparision_btn = gr.Button(value="Run Comparison", variant="primary", scale=-5)
                compare_llm_clear_button = gr.Button(value = "Clear", variant = "secondary", scale = -5)
            
            with gr.Group(visible = False) as compare_comparision_stats:
                compare_comparision_stats_df = gr.Dataframe(label="Comparison Statistics")
                compare_view_comp_btn = gr.Button(value="View Comparison", variant="primary", scale=-5)

            with gr.Group(visible = False) as compare_view_comparision:
                compare_rows_added = gr.Dataframe(label="Rows Added")
                compare_rows_removed = gr.Dataframe(label="Rows Removed")


    compare_run_comparision_btn.click(fn = make_group_visible, inputs = [], outputs = compare_comparision_stats).then(fn = compare_run_file_comparision, inputs = [compare_first_file, compare_first_model, compare_first_version, compare_file_input], outputs = [rows_added, rows_removed, comparision_stats_df])
    compare_view_comp_btn.click(fn = make_group_visible, inputs = [], outputs = compare_view_comparision)

    run_comparision_btn.click(fn = make_group_visible, inputs = [], outputs = comparision_stats).then(fn = run_file_comparision, inputs = [first_file, first_model, first_version, second_file, second_model, second_version], outputs = [rows_added, rows_removed, comparision_stats_df])

    view_comp_btn.click(fn = make_group_visible, inputs = [], outputs = view_comparision)

    file_type.change(fn=toggle_ui, inputs=file_type, outputs=[pdf_ui, excel_ui])
    #add_vendor_btn.click(fn=handle_add_vendor, inputs=[new_vendor_input], outputs=[pdf_vendor_input, excel_vendor_input])

    submit_btn_pdf.click(fn=make_group_visible, inputs=[], outputs=interim_output_group_pdf).then(
        fn=get_interim_output, inputs=[pdf_vendor_input, pdf_file_input, start_page, start_page_table, end_page, appendix_page, appendix_start_table, customization, pdf_model_input,pdf_version_input], outputs=[interim_output_pdf_text]
    ).then(
        fn=make_group_visible, inputs=[], outputs=run_llm_conversion_btn_group_pdf
    ).then(fn = add_pdf_vendor_specific_inputs_to_ddb, inputs = [pdf_vendor_input, start_page, start_page_table, end_page, appendix_page, appendix_start_table, customization], outputs = [])

    run_llm_conversion_btn_pdf.click(fn=make_group_visible, inputs=[], outputs=final_output_group_pdf).then(
        fn=get_final_output, inputs=[pdf_vendor_input,  pdf_file_input, customization, pdf_model_input,pdf_version_input], outputs=final_output_pdf
    ).then(
        fn=make_group_visible, inputs=[], outputs=download_btn_group_pdf
    ).then(fn = handle_comparision_dropdowns, inputs = [pdf_vendor_input,pdf_model_input,pdf_version_input, final_output_pdf], outputs = [first_file, first_model, first_version, second_file, second_model, second_version])

    download_btn_pdf.click(export_csv, [pdf_file_input, final_output_pdf], csv_pdf)



##----------------------------------------------------------------------------------------


    submit_btn_excel.click(fn=make_group_visible, inputs=[], outputs=interim_output_group_excel).then(
        fn=get_interim_output_excel, inputs=[excel_dropdown_inputs[0], excel_file_input, lastrowlist, excel_dropdown_inputs[1], excel_dropdown_inputs[2]], outputs=interim_output_excel_text
    ).then(
        fn=make_group_visible, inputs=[], outputs=run_llm_conversion_btn_group_excel
    ).then(fn = add_excel_vendor_specific_inputs_to_ddb, inputs = [excel_dropdown_inputs[0], lastrowlist, excel_customization], outputs = [])

    run_llm_conversion_btn_excel.click(fn=make_group_visible, inputs=[], outputs=final_output_group_excel).then(
        fn=get_final_output_excel, inputs=[interim_output_excel_text, excel_dropdown_inputs[0], excel_dropdown_inputs[1],excel_dropdown_inputs[2], excel_customization], outputs=final_output_excel
    ).then(
        fn=make_group_visible, inputs=[], outputs=download_btn_group_excel
    ).then(fn = handle_comparision_dropdowns, inputs = [excel_dropdown_inputs[0],excel_dropdown_inputs[1],excel_dropdown_inputs[2], final_output_excel], outputs = [first_file, first_model, first_version, second_file, second_model, second_version])
    
    download_btn_excel.click(export_csv, inputs=[excel_file_input, final_output_excel], outputs=csv_excel)

    clear_button_pdf.click(fn=clear_inputs_pdf, inputs=[], outputs=[
        pdf_vendor_input, pdf_model_input, pdf_version_input, pdf_file_input,
        start_page, start_page_table, end_page, appendix_page, appendix_start_table, customization,
        interim_output_group_pdf, run_llm_conversion_btn_group_pdf, final_output_group_pdf, download_btn_group_pdf
    ])

    clear_button_excel.click(fn=clear_inputs_excel, inputs=[], outputs=[
        excel_dropdown_inputs[0], excel_dropdown_inputs[1], excel_dropdown_inputs[2], excel_file_input, lastrowlist,
        interim_output_excel_text, sheet_names_text, sheet_names_ui, final_output_excel,
        interim_output_group_excel, run_llm_conversion_btn_group_excel, final_output_group_excel, download_btn_group_excel
    ])

    get_sheet_names_button.click(fn=get_sheet_names, inputs=excel_file_input, outputs=sheet_names_text).then(
        fn=make_group_visible, inputs=[], outputs=sheet_names_ui
    )

    pdf_vendor_input.change(fn=handle_custom_vendor, inputs=pdf_vendor_input, outputs=[pdf_vendor_input, excel_dropdown_inputs[0]])
    excel_dropdown_inputs[0].change(fn=handle_custom_vendor, inputs=excel_dropdown_inputs[0], outputs=[pdf_vendor_input, excel_dropdown_inputs[0]])

    pdf_model_input.change(fn=handle_custom_model, inputs=[pdf_vendor_input, pdf_model_input], outputs=[pdf_model_input, excel_dropdown_inputs[1]])
    excel_dropdown_inputs[1].change(fn=handle_custom_model, inputs=[excel_dropdown_inputs[0], excel_dropdown_inputs[1]], outputs=[pdf_model_input, excel_dropdown_inputs[1]])

    pdf_version_input.change(fn=handle_custom_version, inputs=[pdf_vendor_input, pdf_model_input, pdf_version_input], outputs=[pdf_version_input, excel_dropdown_inputs[2]])
    excel_dropdown_inputs[2].change(fn=handle_custom_version, inputs=[excel_dropdown_inputs[0], excel_dropdown_inputs[1], excel_dropdown_inputs[2]], outputs=[pdf_version_input, excel_dropdown_inputs[2]])

    compare_llm_clear_button.click(fn = clear_comparision_tab_llm, inputs = [], outputs = [comparision_stats_df, rows_added, rows_removed, compare_comparision_stats, compare_view_comparision])
    compare_clear_button.click(fn = clear_comparision_tab, inputs = [], outputs = [compare_comparision_stats_df, compare_rows_added, compare_rows_removed, comparision_stats, view_comparision])




interface.launch(share=True, favicon_path="media/cropped-flexgen-icon-32x32.png", server_port=8089)



