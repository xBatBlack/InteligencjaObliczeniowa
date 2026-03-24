import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"c:\Programowanie\InteligencjaObliczeniowa\Projekt03\iris_big.csv")

train_set, test_set = train_test_split(df.values, train_size=0.7, random_state=300852)

print(train_set)

def classify_iris(sl, sw, pl, pw):
    if pl < 2.5:
        return "setosa"
    elif pw > 1.6 or pl > 4.8:
        return "virginica"
    else:
        return "versicolor"

good_predictions = 0
len_test = test_set.shape[0]

for i in range(len_test):
    sl, sw, pl, pw = test_set[i, 0:4]
    true_class = test_set[i, 4]
    
    if classify_iris(sl, sw, pl, pw) == true_class:
        good_predictions += 1

accuracy = (good_predictions / len_test) * 100
print(f"Zgadnięte irysy: {good_predictions} / {len_test}")
print(f"Dokładność (Accuracy): {accuracy:.2f}%")