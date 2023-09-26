import time
from networktables import NetworkTables

# Initialize NetworkTables
NetworkTables.initialize(server='localhost')

# Create a table for testing
table = NetworkTables.getTable('TestTable')

while True:
    # Send some data (e.g., a timestamp)
    timestamp = time.time()
    table.putNumber('Timestamp', timestamp)
    print(f'Sent: Timestamp={timestamp}')
    time.sleep(1)
