import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. Load and prepare the data
data_1 = {
    'buy': ['yes', 'no', 'no', 'yes'],
    'income': ['high', 'high', 'medium', 'low']
}
df_1 = pd.DataFrame(data_1)
print(df_1)

# 2. Encode categorical variables
le_buy = LabelEncoder()
le_income = LabelEncoder()
df_1['buy_encoded'] = le_buy.fit_transform(df_1['buy'])
df_1['income_encoded'] = le_income.fit_transform(df_1['income'])

# 3. Train Naive Bayes model (disable smoothing for exact counts)
model = CategoricalNB(alpha=0)
model.fit(df_1[['income_encoded']], df_1['buy_encoded'])

# 4. Calculate and display prior probabilities
print("Prior Probabilities:")
prior_probs = pd.DataFrame({
    'Class': le_buy.classes_,
    'Probability': np.exp(model.class_log_prior_)
})
print(prior_probs.to_string(index=False))

# 5. Calculate and display conditional probabilities
print("\nConditional Probabilities:")
feature_probs = np.exp(model.feature_log_prob_)

# Reshape based on scikit-learn version
if feature_probs.ndim == 3:
    feature_probs = feature_probs.reshape(2, 3)  # 2 classes × 3 income levels

cond_probs = pd.DataFrame(
    feature_probs,
    columns=le_income.classes_,
    index=le_buy.classes_
)
print(cond_probs)

# 6. Answer the specific probability questions
# P(high|yes)
high_encoded = le_income.transform(['high'])[0]
yes_encoded = le_buy.transform(['yes'])[0]
p_high_given_yes = feature_probs[yes_encoded, high_encoded]
print(f"\n1. Probability that a customer has high income given they bought: P(high|yes) = {p_high_given_yes:.2f}")

# P(yes|high)
X_test = [[high_encoded]]
probs = model.predict_proba(X_test)
p_yes_given_high = probs[0, yes_encoded]
print(f"2. Probability of buying given high income: P(yes|high) = {p_yes_given_high:.2f}")

    
data_2 = {
    'buy': ['yes', 'no', 'no', 'yes'],
    'income': ['high', 'high', 'medium', 'low'],
    'gender': ['male', 'female', 'female', 'male']
}
df_2 = pd.DataFrame(data_2)
print(df_2)

# 2. Encode categorical variables
le_buy = LabelEncoder()
le_income = LabelEncoder()
le_gender = LabelEncoder()

df_2['buy_encoded'] = le_buy.fit_transform(df_2['buy'])
df_2['income_encoded'] = le_income.fit_transform(df_2['income'])
df_2['gender_encoded'] = le_gender.fit_transform(df_2['gender'])

model = CategoricalNB(alpha=0)
model.fit(df_2[['income_encoded', 'gender_encoded']], df_2['buy_encoded'])

# Calculate and display conditional probabilities
print("\nConditional Probabilities:")
print("For Income:")
income_probs = np.exp(model.feature_log_prob_[0])
income_df = pd.DataFrame(
    income_probs,
    columns=le_income.classes_,
    index=le_buy.classes_
)
print(income_df)

print("\nFor Gender:")
gender_probs = np.exp(model.feature_log_prob_[1])
gender_df = pd.DataFrame(
    gender_probs,
    columns=le_gender.classes_,
    index=le_buy.classes_
)
print(gender_df)

# P(high|yes)
high_encoded = le_income.transform(['high'])[0]
yes_encoded = le_buy.transform(['yes'])[0]
p_high_given_yes = income_probs[yes_encoded, high_encoded]
print(f"\n1. Probability that a customer has high income given that he or she bought: P(high|yes) = {p_high_given_yes:.2f}")

# P(male|yes)
male_encoded = le_gender.transform(['male'])[0]
p_male_given_yes = gender_probs[yes_encoded, male_encoded]
print(f"2. Probability that a customer is male given that he bought: P(male|yes) = {p_male_given_yes:.2f}")

# Predict P(yes|high, male)
X_test = [[high_encoded, male_encoded]]
probs = model.predict_proba(X_test)
p_yes_given_high_male = probs[0, yes_encoded]
print(f"\n3. Probability of buying given that customer has high income and is male: P(yes|high,male) = {p_yes_given_high_male:.2f}")