import random

# Define the dataset
news_data = [
    ("The government has passed a new bill on tax reforms.", "Politics"),
    ("The president addressed the nation about the economic plan.", "Politics"),
    ("New policies have been introduced to improve foreign relations.", "Politics"),
    ("A major political party is holding a rally downtown.", "Politics"),
    ("Elections are scheduled to take place next month.", "Politics"),
    ("The hospital has introduced new COVID-19 protocols.", "Health"),
    ("A breakthrough in cancer research has been announced.", "Health"),
    ("Health officials are promoting a new vaccination campaign.", "Health"),
    ("A new diet trend is gaining popularity among health enthusiasts.", "Health"),
    ("Mental health awareness programs are being expanded in schools.", "Health"),
    ("The school district is implementing a new curriculum.", "Education"),
    ("Students are preparing for the national science fair.", "Education"),
    ("A new online learning platform is being launched.", "Education"),
    ("The university is offering new scholarships for undergraduates.", "Education"),
    ("Teachers are receiving training on the latest educational technologies.", "Education"),
    ("The local team won the championship game last night.", "Sports"),
    ("A new world record was set at the Olympics.", "Sports"),
    ("The football season is starting next week.", "Sports"),
    ("A famous athlete announced their retirement today.", "Sports"),
    ("A new sports complex is being built in the city.", "Sports"),
]

# Define the classes
classes = ["Politics", "Health", "Education", "Sports"]

# Random model to predict a class
def random_predict():
    return random.choice(classes)

# Calculate the accuracy
correct_predictions = 0

# Iterate through the dataset and make predictions
for news, true_class in news_data:
    predicted_class = random_predict()
    if predicted_class == true_class:
        correct_predictions += 1

# Calculate the accuracy as a percentage
accuracy = correct_predictions / len(news_data)
accuracy_percentage = accuracy * 100

# Print the accuracy
print(f"Accuracy of the random model: {accuracy_percentage:.2f}%")
