import time
import threading
import logging
from networktables import NetworkTables

# Initialize NetworkTables
NetworkTables.initialize(server='localhost')

# Create a table for testing
table = NetworkTables.getTable('TestTable')

# Function to send data to NetworkTables
def send_data():
    while True:
        # Send some data (e.g., a timestamp)
        timestamp = time.time()
        table.putNumber('Timestamp', 20)
        print(f'Sent: Timestamp={20}')
        time.sleep(1)

# Function to receive data from NetworkTables
def receive_data():
    while True:
        # Get data from NetworkTables
        data = table.getNumber('Timestamp', 0.0)  # 0.0 is the default value
        print(f'Received: Timestamp={data}')
        time.sleep(1)

if __name__ == '__main__':
    # Set up logging to see NetworkTables messages
    logging.basicConfig(level=logging.DEBUG)

    # Create threads for sending and receiving data
    send_thread = threading.Thread(target=send_data)
    # receive_thread = threading.Thread(target=receive_data)

    # Start the threads
    send_thread.start()
    # receive_thread.start()

    # Wait for the threads to finish (you can stop them manually)
    send_thread.join()
    # receive_thread.join()
