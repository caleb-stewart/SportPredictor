import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Define the features names used for training the WHL model
features = ['target_goals', 'opponent_goals', 'target_ppp', 'opponent_ppp', 'target_sog', 'opponent_sog', 'target_fowp', 'opponent_fowp', 'home_away', 'goals_diff', 'ppp_diff', 'sog_diff', 'fowp_diff']
# This is the dependency column we want to predict
target_col = 'target_win'

def train_team_model(team_data):
    '''
    team_data: LIST of dictionaries with features and target column
    Each dictionary should contain the same keys as the features list and the target column.

    This function ONLY TRAINS the model and does not make any predictions.
    It returns the trained model and the scaler used to standardize the features.
    '''

    # Print out the value and type of each feature for debugging
    print("--- Feature values and types for training ---")
    for i, row in enumerate(team_data):
        print(f"Row {i}:")
        for feat in features + [target_col]:
            val = row.get(feat)
            print(f"  {feat}: {val} (type: {type(val)})")
    print("--------------------------------------------")

    # create dataframe from the list of dictionaries
    # appends the dependency column to the features list
    df = pd.DataFrame(team_data, columns=features + [target_col])

    # Throw error if the dependency column is not present
    if target_col not in df.columns:
        raise ValueError(f"Missing {target_col} column.")

    # Get features and depencency columns
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Init scaler and standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define classifiers used
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(random_state=0),
        "SVC": SVC(kernel='rbf', probability=True, random_state=0)
    }

    # Train each classifier on scaled data
    for clf in classifiers.values():
        clf.fit(X_scaled, y)

    # Return the scaler used and trained classifiers
    return {
        "Scaler": scaler,
        "Classifiers": classifiers,
    }


def calculate_team_prob(team_data, models):
    '''
    team_data: LIST of one dictionary with features for the team to predict
    models: Dictionary containing the trained scaler and classifiers

    This function calculates the probability of winning for the team based on the trained models passed in.
    '''
    
    # Make sure the team_data is a DataFrame
    X = pd.DataFrame(team_data, columns=features)

    # Scale the prediction data using the scaler from the trained model
    # We only have one scaler for all classifiers
    X_scaled = models["Scaler"].transform(X)

    # Initialize total probability
    total_prob = 0

    # Iterate through each classifier and get the probability of winning
    for clf in models["Classifiers"].values():
        # [:, 1] is the probability of the positive class (win)
        prob = clf.predict_proba(X_scaled)[:, 1]
        # Add the probability to the total
        total_prob += prob[0]

    # Return the average probability of winning across all classifiers
    return total_prob / len(models["Classifiers"])