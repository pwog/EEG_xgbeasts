from cortex2 import EmotivCortex2Client
import time
import keyboard
import numpy as np
import pandas as pd
import cv2
import time

time_list = []

data = []
url = "wss://localhost:6868"
count_number = 0

    # Start client with authentication
client = EmotivCortex2Client(url,
                            client_id='CLIENT ID',
                            client_secret="CLIENT SECRET",
                            check_response=True,
                            authenticate=True,
                            debug=False)
client.request_access()

client.query_headsets()
client.connect_headset(0)

client.create_activated_session(0)

client.subscribe(streams=["eeg"])


#client.pause_subscriber()
#counter = 0

#while True:
    #print('Please Enter the gesture (0 - ладонь, 1 - кулак)')
    #label = input()
    #if label == 'x':
    #    break
    #for i in range(0, 50):
        #time.sleep(0.1)
#try:
        
data_curr = []

#client.resume_subscriber()
        
        #print(client.receive_data()['time'])
for j in range(20000):
                #data_curr.append([count_number] + client.receive_data()['eeg'][2:16]+[label])
    data_curr.append(client.receive_data()['eeg'][2:16])

        #client.pause_subscriber()

        #data = data + data_curr
data = data_curr
            
        #count_number = count_number + 1
        #if (count_number % 10 == 0):
            #print('--------------------------------------------- 10 Samples were recorded ---------------------------------------------')

    #except:
    #    client.pause_subscriber()
    #   print('--------------------------------------------- Something wrong, no data -----------------------------------------')

df = pd.DataFrame(data)

print('------------------------- DATA SHAPE --------------------- ')
print(df.shape)

print('------------------------------------------DATA SAVED -----------------------------------------------------')
df.to_csv('data_train_new_appr_'+'_label'+str(1)+'.csv', index = False)
    #counter = counter + 1

#client.resume_subscriber()
client.stop_subscriber()
