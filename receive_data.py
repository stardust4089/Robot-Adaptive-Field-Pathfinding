import time
import logging
from networktables import NetworkTables

# Initialize NetworkTables
NetworkTables.initialize(server='localhost')

# Create a table for testing
table = NetworkTables.getTable('TestTable')


# Set up logging to see NetworkTables messages
logging.basicConfig(level=logging.DEBUG)

while True:
    # Get data from NetworkTables
    data = table.getNumber('Timestamp', 0.0)  # 0.0 is the default value
    print(f'Received: Timestamp={data}')
    time.sleep(1)
