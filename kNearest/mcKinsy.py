'''
Created on 10-Mar-2018

@author: Ashwin
'''
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
#from sklearn.neighbors import KNeighborsClassifier

train = pd.read_csv("D:/Downloads/trainMc/train.csv")
test = pd.read_csv("D:/Downloads/testMc/test.csv")
challenge = pd.read_csv("D:/Downloads/trainMc/challenge_data.csv")
b = test.groupby('user_id')['challenge'].apply(list)
frames = [train, test]
joinedDf = pd.concat(frames)
joinedComplete = joinedDf.join(challenge.set_index('challenge_ID'), on='challenge')
testId = test['user_sequence']
trainId = train['user_sequence']
joined_Y = pd.DataFrame(joinedDf['challenge'])
joinedComplete.drop(['author_org_ID'], axis=1, inplace=True)
joinedComplete['author_gender'][joinedComplete.author_gender == 'M'] = int(1)
joinedComplete['author_gender'][joinedComplete.author_gender == 'F'] = int(2)
joinedComplete['author_gender'].fillna(0, inplace=True)
joinedComplete['author_gender'].astype(str, error='ignore').astype(int, error='ignore');
joinedComplete['publish_date'] = pd.to_datetime(joinedComplete['publish_date'], format='%d-%m-%Y');
joinedComplete['publish_date'] = joinedComplete['publish_date'].apply(lambda x: x.strftime('%Y%m'))
joinedComplete['publish_date'] = pd.to_numeric(joinedComplete['publish_date'], downcast='float')
joinedComplete['author_ID'].astype(str);
autIdValueCount = joinedComplete['author_ID'].value_counts();
colList = autIdValueCount[(autIdValueCount > 20000) ].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['author_ID'] == i, 'author_ID'] = 1
colList = autIdValueCount[(autIdValueCount >= 10000) & (autIdValueCount < 20000)  ].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['author_ID'] == i, 'author_ID'] = 2
colList = autIdValueCount[(autIdValueCount >= 5000) & (autIdValueCount < 10000)].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['author_ID'] == i, 'author_ID'] = 3
colList = autIdValueCount[(autIdValueCount >= 1000) & (autIdValueCount < 5000)].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['author_ID'] == i, 'author_ID'] = 4
colList = autIdValueCount[(autIdValueCount >= 500) & (autIdValueCount < 1000) ].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['author_ID'] == i, 'author_ID'] = 5
colList = autIdValueCount[(autIdValueCount >= 100) & (autIdValueCount < 500) ].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['author_ID'] == i, 'author_ID'] = 6
colList = autIdValueCount[(autIdValueCount >= 10) & (autIdValueCount < 100)].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['author_ID'] == i, 'author_ID'] = 7
colList = autIdValueCount[(autIdValueCount < 10) ].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['author_ID'] == i, 'author_ID'] = 8

joinedComplete['author_ID'].fillna(0, inplace=True)
joinedComplete['author_ID'].astype(str).astype(int);
joinedComplete['challenge_series_ID'].fillna(0, inplace=True);
challValueCount = joinedComplete['challenge_series_ID'].value_counts();
colList = challValueCount[(challValueCount > 50000)].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['challenge_series_ID'] == i, 'challenge_series_ID'] = 1
colList = challValueCount[ (challValueCount >= 10000) & (challValueCount < 50000)].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['challenge_series_ID'] == i, 'challenge_series_ID'] = 2
colList = challValueCount[ (challValueCount >= 5000) & (challValueCount < 10000)].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['challenge_series_ID'] == i, 'challenge_series_ID'] = 3
colList = challValueCount[ (challValueCount >= 1000) & (challValueCount < 5000)].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['challenge_series_ID'] == i, 'challenge_series_ID'] = 4
colList = challValueCount[ (challValueCount >= 100) & (challValueCount < 1000)].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['challenge_series_ID'] == i, 'challenge_series_ID'] = 5
colList = challValueCount[ (challValueCount >= 10) & (challValueCount < 100)].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['challenge_series_ID'] == i, 'challenge_series_ID'] = 6
colList = challValueCount[  (challValueCount < 10)].index.tolist()
for i in colList:
    joinedComplete.loc[joinedComplete['challenge_series_ID'] == i, 'challenge_series_ID'] = 7

joinedComplete['challenge_series_ID'].astype(str).astype(int);
joinedComplete['category_id'].fillna(0, inplace=True)
joinedComplete['total_submissions'].fillna(0, inplace=True)
testData = joinedComplete[joinedComplete['user_sequence'].isin(testId)]
testData_seq = testData['user_sequence']
joined_y = joinedComplete['challenge']

joined = joinedComplete.drop(['user_sequence', 'user_id', 'challenge'], axis=1)
testData.drop(['user_sequence', 'user_id', 'challenge'], axis=1, inplace=True)
def predict(X_train, y_train, x_test, k):

    distances = []
    targets = []
    
    for test_i in range(len(x_test)):
        for i in range(len(X_train)):

            distance = np.sqrt(np.sum(np.square(x_test[test_i, :] - X_train[i, :])))

            distances.append([distance, i])

    distances = sorted(distances)


    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])


    return targets
def kNearestNeighbor(X_train, y_train, X_test, predictions, k):

# loop over all observations
    for i in range(0, len(X_test), 10):
        predictions.append(predict(X_train, y_train, X_test[i: i + 10, :], k))
predictions = []
kNearestNeighbor(joined.as_matrix(), joined_y.as_matrix(), testData.as_matrix(), predictions, 13)

# transform the list into an array
predictions = np.asarray(predictions)
