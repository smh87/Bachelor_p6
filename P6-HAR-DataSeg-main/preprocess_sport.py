import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


from dotenv import load_dotenv
def preprocess_sport():
    load_dotenv()
    filepath = os.getenv('DATA_SPORT')
    print(filepath)

    processedList = []
    def processLines():
        lines = file.readlines()[1:]
        for i, line in enumerate(lines):
            try:
                line = line.split(',')
                for i in range(len(line)):
                    if line[i] == '':
                        line[i] = '0'
                        line[i].replace('\n', '')
                        line[9]=line[9].strip()
                temp = [line[0], line[1], line[2], line[3], line[4], line[5], line[6], line[7], line[8], line[9]]
                if line[0] is not (''):
                    processedList.append(temp)
            except:
                print('Error at line number: ', i)


    for i in range (1, 10):
        name = "{i}badminton.txt".format(i=i)
        File1 = os.path.join(filepath + "\\badminton\\" + name)
        file = open(File1)
        processLines()    
    columns = ['activity','feature0','feature1','feature2',
    'feature3','feature4','feature5',
    'feature6','feature7','feature8']
    data_temp = pd.DataFrame(data = processedList, columns = columns)
    data_temp2 = data_temp.drop(columns=['activity'])
    data_temp2.insert(0,'activity','badminton')
    data_badminton=data_temp2.head(7000).copy()
    processedList.clear()

    for i in range(1, 12):
        name = "{i}basketball.txt".format(i=i)
        File1 = os.path.join(filepath + "\\basketball\\" + name)
        file = open(File1)
        processLines()
    columns = ['activity','feature0','feature1','feature2',
    'feature3','feature4','feature5',
    'feature6','feature7','feature8']
    data_tempbasket = pd.DataFrame(data = processedList, columns = columns)
    data_temp2basket = data_tempbasket.drop(columns=['activity'])
    data_temp2basket.insert(0,'activity','basketball')
    data_basketball=data_temp2basket.head(7000).copy()
    processedList.clear()
        

    for i in range (1, 17):
        name = "{i}cycling.txt".format(i=i)
        File1 = os.path.join(filepath + "\\cycling\\" + name)
        file = open(File1)
        processLines()
    columns = ['activity','feature0','feature1','feature2',
    'feature3','feature4','feature5',
    'feature6','feature7','feature8']
    data_tempcycle = pd.DataFrame(data = processedList, columns = columns)
    data_temp2cycle = data_tempcycle.drop(columns=['activity'])
    data_temp2cycle.insert(0,'activity','cycle')
    data_cycle = data_temp2cycle.head(7000).copy()
    processedList.clear()

    for i in range (1, 11):
        name = "{i}football.txt".format(i=i)
        File1 = os.path.join(filepath + "\\football\\" + name)
        file = open(File1)
        processLines()
    columns = ['activity','feature0','feature1','feature2',
    'feature3','feature4','feature5',
    'feature6','feature7','feature8']
    data_tempfootball = pd.DataFrame(data = processedList, columns = columns)
    data_temp2football = data_tempfootball.drop(columns=['activity'])
    data_temp2football.insert(0,'activity','football')
    data_football = data_temp2football.copy()
    processedList.clear()

    for i in range (1, 20):
        name = "{i}skipping.txt".format(i=i)
        File1 = os.path.join(filepath + "\\skipping\\" + name)
        file = open(File1)
        processLines()
    columns = ['activity','feature0','feature1','feature2',
    'feature3','feature4','feature5',
    'feature6','feature7','feature8']
    data_tempskipping = pd.DataFrame(data = processedList, columns = columns)
    data_temp2skipping = data_tempskipping.drop(columns=['activity'])
    data_temp2skipping.insert(0,'activity','skipping')
    data_skipping = data_temp2skipping.head(7000).copy()
    processedList.clear()

    for i in range (1, 15):
        name = "{i}tabletennis.txt".format(i=i)
        File1 = os.path.join(filepath + "\\tabletennis\\" + name)
        file = open(File1)
        processLines()
    columns = ['activity','feature0','feature1','feature2',
    'feature3','feature4','feature5',
    'feature6','feature7','feature8']
    data_temptabletennis = pd.DataFrame(data = processedList, columns = columns)
    data_temp2tabletennis = data_temptabletennis.drop(columns=['activity'])
    data_temp2tabletennis.insert(0,'activity','tabletennis')
    data_tabletennis = data_temp2tabletennis.head(7000).copy()
    processedList.clear()

    # joining the activity tables
    frames = [data_badminton, data_basketball, data_cycle, data_football, data_skipping, data_tabletennis]

    datafinal = pd.concat(frames,ignore_index=True)
    datasuper = pd.DataFrame(data = datafinal, columns=columns)
    #datasuper = datasuper.replace('\n','', regex=True)
    for i in range(9):
        try:
            datasuper[('feature'+str(i))] = datasuper[('feature'+str(i))].astype('float')
        except:
            print(datasuper[('feature'+str(i))].values)
            exit()


    label = LabelEncoder()
    datasuper['label'] = label.fit_transform(datasuper['activity'])
    #datasuper.head()

    X = datasuper[['feature0','feature1','feature2',
    'feature3','feature4','feature5',
    'feature6','feature7','feature8']]
    y = datasuper['label']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    scaled_X = pd.DataFrame(data = X, columns = ['feature0','feature1','feature2',
    'feature3','feature4','feature5',
    'feature6','feature7','feature8'])
    scaled_X['label'] = y.values

    print(scaled_X, label)
    return scaled_X, label
