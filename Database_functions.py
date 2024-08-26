from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import boto3
import logging
import uuid
from boto3.dynamodb.conditions import Attr



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
dynamodb = boto3.resource('dynamodb', region_name='us-east-1', aws_access_key_id='AKIA6KL2BTNPZ7LGCFAP', aws_secret_access_key='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')

main_table = 'modbus_reg_ex'
processed_table = 'FlexGen_Processed_Vendors'

def fetch_vendors_from_dynamodb():
    logging.debug("Fetching vendors from DynamoDB")
    try:
        table = dynamodb.Table(main_table)
        response = table.scan(
        FilterExpression=Attr('pk').eq('Vendor')
        )
        # Extract the vendor names from the response
        vendors = [item['name'] for item in response['Items']]
        logging.info(f"Fetched vendors: {vendors}")
        return vendors
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error fetching data from DynamoDB: {str(e)}")
        return ["Error fetching data from DynamoDB. Check your AWS credentials."]

def add_vendor_to_dynamodb(new_vendor):
    logging.debug(f"Adding vendor {new_vendor} to DynamoDB")
    try:
        table = dynamodb.Table(main_table)
        vendor_id = str(uuid.uuid4())
        
        # First write
        table.put_item(Item={
            'pk': 'Vendor',
            'sk': f'{vendor_id}',
            'name': new_vendor
        })
        
        # Second write
        table.put_item(Item={
            'pk': f'Vendor:{vendor_id}',
            'sk': f'Vendor:{vendor_id}',
            'name': new_vendor
        })
        
        logging.info(f"Added new vendor: {new_vendor}")
        return fetch_vendors_from_dynamodb()
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error adding data to DynamoDB: {str(e)}")
        return ["Error adding data to DynamoDB. Check your AWS credentials."]
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return ["Unexpected error occurred. Check the logs for more details."]

def fetch_versions_from_dynamodb():
    logging.debug("Fetching versions from DynamoDB")
    try:
        table = dynamodb.Table(main_table)
        # Scan the table with a filter condition on sk to get items where sk begins with 'Model'
        response = table.scan(
            FilterExpression=Attr('sk').begins_with('Model')
        )
        
        # Dictionary to store unique versions for each model
        model_versions = {}
        
        for item in response['Items']:
            model_name = item['name']
            versions = item.get('versions', [])

            # Ensure versions is a list of strings
            if isinstance(versions, list):
                version_list = []
                for version in versions:
                    if isinstance(version, dict) and 'S' in version:
                        version_list.append(version['S'])
                    elif isinstance(version, str):
                        version_list.append(version)
            else:
                version_list = []

            if model_name in model_versions:
                model_versions[model_name].update(version_list)
            else:
                model_versions[model_name] = set(version_list)
        
        # Convert sets to lists for easier use
        for model_name in model_versions:
            model_versions[model_name] = list(model_versions[model_name])
        
        return model_versions
    
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error fetching data from DynamoDB: {str(e)}")
        return ["Error fetching data from DynamoDB. Check your AWS credentials."]

def add_versions_to_dynamodb(vendor_name, model_name, new_version):
    logging.debug(f"Adding version '{new_version}' for model '{model_name}' under vendor '{vendor_name}'")
    try:
        table = dynamodb.Table(main_table)

        # Fetch the vendor item to get the vendor ID (sk for the vendor)
        vendor_response = table.scan(
            FilterExpression=Attr('pk').eq('Vendor') & Attr('name').eq(vendor_name)
        )
        vendor_items = vendor_response.get('Items', [])
        if not vendor_items:
            logging.error(f"Vendor '{vendor_name}' not found.")
            return

        vendor_item = vendor_items[0]
        vendor_sk = vendor_item.get('sk')

        # Fetch the model item under the vendor
        model_response = table.scan(
            FilterExpression=Attr('pk').eq(f'Vendor:{vendor_sk}') & Attr('sk').begins_with('Model') & Attr('name').eq(model_name)
        )
        model_items = model_response.get('Items', [])
        
        if model_items:
            # If model exists, update it with the new version
            model_item = model_items[0]
            pk = model_item['pk']
            sk = model_item['sk']
            current_versions = model_item.get('versions', [])

            # Check if the version is already present
            if any(ver.get('S') == new_version for ver in current_versions):
                logging.info(f"Version '{new_version}' already exists for model '{model_name}'.")
                return

            # Add the new version
            current_versions.append({'S': new_version})

            # Update the item in the table with the new version list
            table.update_item(
                Key={
                    'pk': pk,
                    'sk': sk
                },
                UpdateExpression='SET versions = :new_versions',
                ExpressionAttributeValues={
                    ':new_versions': current_versions
                }
            )
            logging.info(f"Updated model '{model_name}' with new version '{new_version}' under vendor '{vendor_name}'.")

        else:
            # If model does not exist, create a new item with the new version
            model_id = str(uuid.uuid4())
            new_item = {
                'pk': f'Vendor:{vendor_sk}',
                'sk': f'Model:{model_id}',
                'name': model_name,
                'versions': [{'S': new_version}]
            }
            table.put_item(Item=new_item)
            logging.info(f"Added new model '{model_name}' with version '{new_version}' under vendor '{vendor_name}'.")

        model_versions = fetch_versions_from_dynamodb()
        models = list(model_versions.keys())
        # Collect unique versions
        unique_versions = set()
        for model, versions in model_versions.items():
            unique_versions.update(versions)
            print(f"Model: {model}, Versions: {versions}")

        # Convert the set of unique versions to a list
        versions = list(unique_versions)
        return versions

    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error adding version to DynamoDB: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")


def fetch_models_from_dynamodb():
    logging.debug("Fetching versions from DynamoDB")
    try:
        table = dynamodb.Table(main_table)
        # response = table.scan()
        # items = response.get('Items', [])
        # models = [item['Model_Name'] for item in items]
        response = table.scan(FilterExpression=Attr('sk').begins_with('Model'))
        # Extract the model names from the response
        models = list({item['name'] for item in response['Items']})
        logging.info(f"Fetched Models: {models}")
        return models
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error fetching data from DynamoDB: {str(e)}")
        return ["Error fetching data from DynamoDB. Check your AWS credentials."]



def add_models_to_dynamodb(vendor_name, new_model):
    logging.debug("Adding model to DynamoDB")
    try:
        table = dynamodb.Table(main_table)
        
        # Scan the table to find the vendor's item
        response = table.scan(
            FilterExpression=Attr('pk').begins_with('Vendor') & Attr('name').eq(vendor_name)
        )

        # Extract the SK from the response
        items = response.get('Items', [])
        if not items:
            logging.error(f"Vendor '{vendor_name}' not found.")
            return ["Vendor not found."]

        item = items[0]
        sk = item.get('sk')
        
        # Generate a new unique model ID
        model_id = str(uuid.uuid4())
        
        # Insert new model item into the table
        table.put_item(Item={
            'pk': sk,
            'sk': f'Model:{model_id}',
            'name': new_model
        })
        
        logging.info(f"Added new model: {new_model}")
        return fetch_models_from_dynamodb()
    
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error adding data to DynamoDB: {str(e)}")
        return ["Error adding data to DynamoDB. Check your AWS credentials."]
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return ["Unexpected error occurred. Check the logs for more details."]



def add_processed_vendors_to_ddb(vendor,model,version, df_size):
    logging.debug(f"Adding vendor {vendor} to DynamoDB along with model {model} and version {version}")
    try:
        table = dynamodb.Table(processed_table)        
        # First write
        table.put_item(Item={
            'vendor': vendor,
            'model': model,
            'version': version,
            'row_count':df_size
        })
        return fetch_comparision_vendors(), fetch_comparision_models(), fetch_comparision_versions()
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error adding data to DynamoDB: {str(e)}")
        return ["Error adding data to DynamoDB. Check your AWS credentials."]
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return ["Unexpected error occurred. Check the logs for more details."]
    
def add_pdf_vendor_specific_inputs_to_ddb(pdf_vendor_input, start_page, start_page_table, end_page, appendix_page, appendix_start_table, customization):
    logging.debug("Adding vendor specific details to DynamoDB for PDF vendors")
    user_inputs = [start_page, start_page_table, end_page, appendix_page, appendix_start_table]
    table = dynamodb.Table('modbus_reg_ex')

    try:
        # Query the table to find the vendor's item
        response = table.scan(
            FilterExpression=Attr('pk').eq('Vendor') & Attr('name').eq(pdf_vendor_input)
        )

        items = response.get('Items', [])
        if not items:
            logging.error(f"Vendor '{pdf_vendor_input}' not found.")
            return ["Vendor not found."]

        item = items[0]
        vendor_sk = item.get('sk')

        # Insert the specific input details into the table
        table.put_item(Item={
            'pk': f'Vendor:{vendor_sk}',
            'sk': f'Input:{uuid.uuid4()}',  # Assuming a new unique sk for each input
            'name': pdf_vendor_input,
            'User_inputs': user_inputs,
            'File_type': 'pdf',
            'Customization': customization
        })

        logging.info(f"Added PDF vendor specific inputs for '{pdf_vendor_input}' with sk '{vendor_sk}'.")
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error adding data to DynamoDB: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")


def add_excel_vendor_specific_inputs_to_ddb(excel_vendor_input, last_rows, customization):
    logging.debug("Adding vendor specific details to dynamo db for excel vendors")
    table = dynamodb.Table('modbus_reg_ex')

    # Query the table to find the vendor's item
    try:
        response = table.scan(
            FilterExpression=Attr('pk').eq('Vendor') & Attr('name').eq(excel_vendor_input)
        )
        items = response.get('Items', [])
        if not items:
            logging.error(f"Vendor '{excel_vendor_input}' not found.")
            return ["Vendor not found."]
        
        item = items[0]
        vendor_sk = item.get('sk')

        # Insert the specific input details into the table
        table.put_item(Item={
            'pk': f'Vendor:{vendor_sk}',
            'sk': f'Input:{uuid.uuid4()}',  # Assuming a new unique sk for each input
            'File_type': 'excel',
            'name': excel_vendor_input,
            'User_inputs': last_rows,
            'Customization': customization
        })

        logging.info(f"Added vendor specific inputs for '{excel_vendor_input}' with sk '{vendor_sk}'.")
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error adding data to DynamoDB: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")

def fetch_comparision_vendors():
    logging.debug("Fetching vendors from Processed DynamoDB Table")
    try:
        table = dynamodb.Table(processed_table)
        response = table.scan()
        items = response.get('Items', [])
        vendors = [item['vendor'] for item in items]
        return vendors
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error adding data to DynamoDB: {str(e)}")
        return ["Error adding data to DynamoDB. Check your AWS credentials."]
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return ["Unexpected error occurred. Check the logs for more details."]
    
def fetch_comparision_models():
    logging.debug("Fetching model from Processed DynamoDB Table")
    try:
        table = dynamodb.Table(processed_table)
        response = table.scan()
        items = response.get('Items', [])
        models = [item['model'] for item in items]
        return models
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error adding data to DynamoDB: {str(e)}")
        return ["Error adding data to DynamoDB. Check your AWS credentials."]
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return ["Unexpected error occurred. Check the logs for more details."]
    
def fetch_comparision_versions():
    logging.debug("Fetching version from Processed DynamoDB Table")
    try:
        table = dynamodb.Table(processed_table)
        response = table.scan()
        items = response.get('Items', [])
        versions = [item['version'] for item in items]
        return versions
    except (NoCredentialsError, PartialCredentialsError) as e:
        logging.error(f"Error adding data to DynamoDB: {str(e)}")
        return ["Error adding data to DynamoDB. Check your AWS credentials."]
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return ["Unexpected error occurred. Check the logs for more details."]
    

