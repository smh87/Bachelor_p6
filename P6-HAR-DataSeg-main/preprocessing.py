import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


from dotenv import load_dotenv
def preprocess():
    load_dotenv()
    filepath = os.getenv('DATA_PATH')
    print(filepath)
    file = open(filepath)

    lines = file.readlines()

    processedList = []

    for i, line in enumerate(lines):
        try:
            line = line.split(',')
            if len(line) <= 6:
                last = line[5].split(';')[0]
                last.strip()
            else:
                last = line[5]
            if last == '' or last== ' ' or last == '\n':
                print(i)
                break
            for li in [line[0], line[1], line[2], line[3], line[4], last]:
                if li.find('\n') > 0:
                    print('wtf at ', i)
            temp = [line[0], line[1], line[2], line[3], line[4], last]
            
            processedList.append(temp)
        except Exception:
            print('Error at line number: ', i)


    columns = ['user', 'activity', 'time', 'feature0', 'feature1', 'feature2']
    data = pd.DataFrame(data = processedList, columns = columns)
    print(set(data['user']))
    #print(data.head())
    #print(data.info())
    #print(data.isnull().sum())

    data['feature0'] = data['feature0'].astype('float')
    data['feature1'] = data['feature1'].astype('float')
    data['feature2'] = data['feature2'].astype('float')
    


    Fs = 20
    activities = data['activity'].value_counts().index
    # def plot_activity(activity, data):
    #     fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(15, 7), sharex=True)
    #     plot_axis(ax0, data['time'], data['x'], 'X-Axis')
    #     plot_axis(ax1, data['time'], data['y'], 'Y-Axis')
    #     plot_axis(ax2, data['time'], data['z'], 'Z-Axis')
    #     plt.subplots_adjust(hspace=0.2)
    #     fig.suptitle(activity)
    #     plt.subplots_adjust(top=0.90)
    #     plt.show()

    # def plot_axis(ax, x, y, title):
    #     ax.plot(x, y, 'g')
    #     ax.set_title(title)
    #     ax.xaxis.set_visible(False)
    #     ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    #     ax.set_xlim([min(x), max(x)])
    #     ax.grid(True)

    # for activity in activities:
    #     data_for_plot = data[(data['activity'] == activity)][:Fs*10]
    #     plot_activity(activity, data_for_plot)
        
    df = data.drop(['time'], axis = 1).copy()
    df.head()
    X = []
    label = LabelEncoder()
    label.fit(list(set(df['activity'])))
    for user in set(df['user']):
        user_frame = df[df['user'] == user].copy()
        Walking = user_frame[user_frame['activity']=='Walking'].copy()
        Jogging = user_frame[user_frame['activity']=='Jogging'].copy()
        Upstairs = user_frame[user_frame['activity']=='Upstairs'].copy()
        Downstairs = user_frame[user_frame['activity']=='Downstairs'].copy()
        Sitting = user_frame[user_frame['activity']=='Sitting'].copy()
        Standing = user_frame[user_frame['activity']=='Standing'].copy()

        balanced_data = pd.DataFrame()
        balanced_data = balanced_data.append([Walking, Jogging, Upstairs, Downstairs, Sitting, Standing])
        balanced_data.shape


        
        balanced_data['label'] = label.transform(balanced_data['activity'])
        balanced_data.head()

        #Standardize data
        scaler = StandardScaler()
        temp_X = balanced_data[['feature0', 'feature1', 'feature2']]
        temp_X = scaler.fit_transform(temp_X)
        temp_X = pd.DataFrame(data = temp_X, columns = ['feature0', 'feature1', 'feature2'])
        y = balanced_data['label']
        temp_X['label'] = y.values
        
        X.append(temp_X)
        

    return X, label

if __name__ == '__main__':
    X, label = preprocess()
    print(X)
    print(len(X))
    #print(X, label)