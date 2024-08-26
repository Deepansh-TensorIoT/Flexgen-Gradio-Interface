import logging

# Configure the main logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set minimum level to capture

# Create a handler that will print the logs to stdout
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create a formatter and set it to the handler
formatter = logging.Formatter('%(levelname)s: %(message)s')
stream_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(stream_handler)

# Suppress third-party library logs by setting their log level higher
# logging.getLogger("boto3").setLevel(logging.WARNING)
# logging.getLogger("botocore").setLevel(logging.WARNING)
# logging.getLogger("urllib3").setLevel(logging.WARNING)
# logging.getLogger("httpcore").setLevel(logging.WARNING)
# logging.getLogger("PIL").setLevel(logging.WARNING)
# Add other third-party libraries as needed