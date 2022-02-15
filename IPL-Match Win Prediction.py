import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
deliveries_df=pd.read_csv("deliveries.csv")
match_df=pd.read_csv("matches.csv")
#print(match_df.to_string())
#print(deliveries_df.to_string())
#  --Step2--
# Analyze the data types and number of null values in both the CSV files.
#print(match_df.info())
#print(deliveries_df.info())

# If any respective row in winner column is null it is considered as a Draw Match
match_df["winner"].fillna("Draw",inplace=True)
print(match_df.info())

# For columns having team names
# Label Encoding mechanism is manually done for columns having Team-Names
# Because Rising Pune-Super-giant team exist with 2 different names throughout the csv file which cant be
# rectified by LabelEncoder
team_encoding={
    "Mumbai Indians":1,
    "Kolkata Knight Riders":2,
    "Royal Challengers Bangalore":3,
    "Deccan Chargers":4,
    "Chennai Super Kings":5,
    "Rajasthan Royals":6,
    "Delhi Daredevils":7,
    "Gujarat Lions":8,
    "Kings XI Punjab":9,
    "Sunrisers Hyderabad":10,
    "Rising Pune Supergiants":11,
    "Rising Pune Supergiant":11,
    "Kochi Tuskers Kerala":12,
    "Pune Warriors":13,
    "Delhi Capitals":14,
    "Draw":15
}
team_encode_dict={
    "team1":team_encoding,
    "team2":team_encoding,
    "toss_winner":team_encoding,
    "winner":team_encoding
}

# for City column

match_df.replace(team_encode_dict,inplace=True)
#print(match_df.to_string())
# -- Replace NAN values of City column
print("City Analysis : ",match_df[match_df["city"].isnull()==True].to_string())
# From the above null columns for all city having null values the venue is Dubai International stadium
# Hence replace null columns with Dubai as city
match_df["city"].fillna("Dubai",inplace=True)
print(match_df.info())

#

# Lets consider toss wins and match wins by a team
toss_wins=match_df["toss_winner"].value_counts(sort=True)
match_wins=match_df["winner"].value_counts(sort=True)
print("No of Toss Wins : ")
for id,val in toss_wins.iteritems():
    print(f"{list(team_encode_dict['winner'].keys())[id-1]} ={toss_wins[id]}")
print("No of Match Wins : ")
for id,val in match_wins.iteritems():
    print(f"{list(team_encode_dict['winner'].keys())[id-1]} ={match_wins[id]}")
# Plot a BAR-graph between Toss winners and match Winners to know if there is any relation
fig=plt.figure(figsize=(8,4))
ax1=fig.add_subplot(121)
ax1.set_xlabel("Team")
ax1.set_ylabel("Count of Toss Wins")
ax1.set_title("Toss Winners")
(toss_wins.plot(kind="bar"))
ax2=fig.add_subplot(122)
(match_wins.plot(kind="bar"))
ax2.set_xlabel("Team")
ax2.set_ylabel("Count of Matches Won")
ax2.set_title("Match Winners")
plt.show()


# Consider columns having no or very less redundancy
match_df=match_df[["team1","team2","venue","city","toss_decision","toss_winner","winner"]]
print(match_df.info())


# Convert String values of respective columns to Integer using LabelEncoder
x=["city","toss_decision","venue"]
lb=LabelEncoder()
for i in x:
    match_df[i]=lb.fit_transform(match_df[i])
print(match_df.head())

# Start Training the model
train_data,test_data=train_test_split(match_df,test_size=0.20,random_state=5)
print(train_data.shape)
print(test_data.shape)
def print_model_scores(model,data,predictor,target):
    model.fit(data[predictor],data[target])
    predictions=model.predict(data[predictor])
    accuracy=accuracy_score(predictions,data[target])
    print("Accuracy %s" % "{0:.2}".format(accuracy))
    scores=cross_val_score(model,data[predictor],data[target],scoring="neg_mean_squared_error",cv=5)
    print("Cross-validation scores : {}".format(np.sqrt(-scores)))
    print("Avarage RMSE : ",np.sqrt(-scores).mean())
target_col=["winner"]
predictor_col=["team1","team2","venue","toss_winner","city","toss_decision"]
#model=LogisticRegression()
#print_model_scores(model,train_data,predictor_col,target_col)
model=RandomForestClassifier(n_estimators=100)
print_model_scores(model,train_data,predictor_col,target_col)
team1=input("Enter team1 Name : ")
team2=input("Enter team2 Name : ")
toss_winners=input("Enter who won the toss : ")
inpt=[team_encode_dict["team1"][team1],team_encode_dict['team2'][team2],'14',team_encode_dict['toss_winner'][toss_winners],'2','1']
print(inpt)
inpt=np.array(inpt).reshape((1,-1))
print(inpt)
output=model.predict(inpt)
print(output)
print(f"The Winner would be : {list(team_encoding.keys())[list(team_encode_dict['team1'].values()).index(output)]}")







