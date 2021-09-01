import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dotenv import load_dotenv

def preprocess():
    load_dotenv()
    filepath = os.getenv('MHEALTH')
    print(filepath)


    processedList = []
    all_activities = ['Sitting', 'Standing', 'Lying on back', 'Lying on right side', 'Ascending stairs', 'Descending stairs', 'Standing in elevator', 'Moving in elevator', 'Walking on parking lot', 'Walking on flat treadmill', 'Walking on inclined treadmill', 'Running on treadmill', 'Exercising on stepper', 'Exercising on cross trainer', 'Cycling horizontal', 'Cycling vertical', 'Rowing', 'Jumping', 'Playing basketball']
    all_features= []
    for i in range(45):
        all_features.append(f'feature{i}')
    all_features_and_activity = all_features.copy()
    all_features_and_activity.append('activity')
    

    for i in range(8):
        processedList.append([])

    for activity_nr in range(1, 20):
        folder = ''
        if activity_nr < 10:
            folder = f'a0{activity_nr}'
        else:
            folder = f'a{activity_nr}'
        for participant in range (1, 9):
            for segment_nr in range(1, 61):
                segment = ''
                if segment_nr < 10:
                    segment = f's0{segment_nr}.txt'
                else:
                    segment = f's{segment_nr}.txt'
                file_name = f'{filepath}/{folder}/p{participant}/{segment}'
                with open(file_name,'r') as file:
                    for i, line in enumerate(file.readlines()):
                        # try:
                        line = line.split(',')
                        line[-1] = line[-1].split('\n')[0]
                        line.append(all_activities[activity_nr-1])
                        processedList[participant-1].append(line)
                        # except Exception as e:
                        #     print('activity ', activity_nr)
                        #     print('participant ', participant)
                        #     print('file ', segment_nr)
                        #     print('Error at line number: ', i)
                        #     print(e)

                

    # for i, line in enumerate(file.readlines()):
    #     try:
    #         line = line.split(',')
    #         last = line[7].split('\n')[0]
    #         for li in [line[0], line[1], line[2], line[3], line[4], line[5], line[6], last]:
    #             if li.find('\n') > 0:
    #                 print('wtf at ', i)
    #         user = ord(line[0][0]) - ord('A')
    #         temp = [user, line[1], line[2], line[3], line[4],  line[5], line[6], last]
            
    #         processedList.append(temp)
    #     except Exception:
    #         print('Error at line number: ', i)
    all_data = []
    for frame in processedList:
        data = pd.DataFrame(data = frame, columns = all_features_and_activity)
        for i in range(45):
            data[f'feature{i}'] = data[f'feature{i}'].astype('float')
        all_data.append(data)

    



    df = all_data.copy()
    X = []
    label = LabelEncoder()
    label.fit(list(set(df[0]['activity'])))
    for frame in df:
        

        balanced_data = pd.DataFrame()
        balanced_data = balanced_data.append(frame)
        balanced_data.shape


        
        balanced_data['label'] = label.transform(balanced_data['activity'])
        balanced_data.head()

        #Standardize data
        scaler = StandardScaler()
        temp_X = balanced_data[all_features]
        temp_X = scaler.fit_transform(temp_X)
        temp_X = pd.DataFrame(data = temp_X, columns = all_features)
        y = balanced_data['label']
        temp_X['label'] = y.values
        
        X.append(temp_X)
        

    return X, label

if __name__ == '__main__':
    X, label = preprocess()
    print(len(X))
    #print(X, label)
