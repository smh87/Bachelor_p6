import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


from dotenv import load_dotenv
def preprocess2():
    load_dotenv()
    filepath = os.getenv('DATA_IMSHA')
    ##print(filepath)
    #user defined columns
    columns = ['activity','feature0','feature1','feature2',
    'feature3','feature4','feature5',
    'feature6','feature7','feature8',
    'feature9','feature10','feature11',
    'feature12','feature13','feature14',
    'feature15','feature16','feature17',
    'feature18','feature19','feature20',
    'feature21','feature22','feature23',
    'feature24','feature25','feature26']
    #list to save the dataframes
    read_dataframes = {}
    #read subject csv files as dataframe
    for root, dirs, files in os.walk(filepath):
        for names in files:
            #read subject 1-9
            if names.find("imu")!=-1 and root[-1]!=str(0):
                read_dataframes[f"subject_{root[-1]}"] = pd.read_csv(os.path.join(root,names), sep=',',skiprows=lambda x: x in [0], skip_blank_lines=True, error_bad_lines=True, names=columns)
            elif root[-1]==str(0):
                #read subject 10
                read_dataframes[f"subject_{root[-2]+root[-1]}"] = pd.read_csv(os.path.join(root,names), sep=',',skiprows=lambda x: x in [0], skip_blank_lines=True, error_bad_lines=True, names=columns)
    #float convert 
    for i in range(27):
        for j in range(1,11):
            read_dataframes[f'subject_{j}'][('feature'+str(i))] = read_dataframes[f'subject_{j}'][('feature'+str(i))].astype('float')
    
    #label encoder 
    label = LabelEncoder()
    
    #rename activities and drop empty missing or NaN values
    for j in range(1,11):
        read_dataframes[f'subject_{j}']['activity'] = read_dataframes[f'subject_{j}']['activity'].replace(1,'Using Computer')
        read_dataframes[f'subject_{j}']['activity'] = read_dataframes[f'subject_{j}']['activity'].replace(2, 'Phone Conversation')
        read_dataframes[f'subject_{j}']['activity'] = read_dataframes[f'subject_{j}']['activity'].replace(3, 'Vaccum Cleaning')
        read_dataframes[f'subject_{j}']['activity'] = read_dataframes[f'subject_{j}']['activity'].replace(4, 'Reading Book')
        read_dataframes[f'subject_{j}']['activity'] = read_dataframes[f'subject_{j}']['activity'].replace(5, 'Watching TV')
        read_dataframes[f'subject_{j}']['activity'] = read_dataframes[f'subject_{j}']['activity'].replace(6, 'Ironing')
        read_dataframes[f'subject_{j}']['activity'] = read_dataframes[f'subject_{j}']['activity'].replace(7, 'Walking')
        read_dataframes[f'subject_{j}']['activity'] = read_dataframes[f'subject_{j}']['activity'].replace(8, 'Exercise')
        read_dataframes[f'subject_{j}']['activity'] = read_dataframes[f'subject_{j}']['activity'].replace(9, 'Cooking')
        read_dataframes[f'subject_{j}']['activity'] = read_dataframes[f'subject_{j}']['activity'].replace(10, 'Drinking')
        read_dataframes[f'subject_{j}']['activity'] = read_dataframes[f'subject_{j}']['activity'].replace(11, 'Brushing Hair')
        #drop missing values rows
        read_dataframes[f'subject_{j}'].dropna(axis=0, inplace=True)
        read_dataframes[f'subject_{j}'].shape
    #Fit label encoder
    temp_acitivitylist = ['Cooking', 'Using Computer', 'Reading Book', 'Watching TV', 'Phone Conversation', 'Vaccum Cleaning', 'Exercise', 'Ironing', 'Brushing Hair', 'Drinking', 'Walking']
    label.fit(list(set(temp_acitivitylist)))

    ##list X each activity as value where values are dataframes
    X = []
    
    #
    for j in range(1,11):
        #print("\n index " + f"{j}")
        #Standardize data
        scaler = StandardScaler()
        ##print("\n index " + f"{j}")
        
        #Fit label encoder and return encoded labels        
        read_dataframes[f"subject_{j}"]['label'] = label.transform(read_dataframes[f"subject_{j}"]['activity'])
        #print(read_dataframes[f"subject_{j}"].head())

        #Standardize data
        scaler = StandardScaler()
        temp_X = pd.DataFrame()
        temp_X = read_dataframes[f"subject_{j}"][['feature0','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8',
    'feature9','feature10','feature11',
    'feature12','feature13','feature14',
    'feature15','feature16','feature17',
    'feature18','feature19','feature20',
    'feature21','feature22','feature23',
    'feature24','feature25','feature26']].copy()
        scaler.fit_transform(temp_X)
        y = read_dataframes[f"subject_{j}"]['label']
        temp_X['label'] = y.values
        X.append(temp_X)
    ##print(len(X))
    ##print(scaled_X, label)
    return X, label

#scaled_X_test, label_test= preprocess2()
##print(len(scaled_X_test))
