# Simple Chatbot with Persistent Memory (MySQL) - Flan-T5 (NOT FINE-TUNED)

This is a simple chatbot built using Streamlit, Flan-T5, and MySQL for persistent memory. The chatbot can detect user intent, respond accordingly, analyze sentiment, and summarize long conversations.

## Features
- **Natural Language Processing (NLP):** Uses Flan-T5 for generating responses.
- **Intent Detection:** Classifies user queries into predefined categories.
- **Sentiment Analysis:** Determines whether the user's input is positive or negative.
- **Conversation Summarization:** Condenses long conversations when needed.
- **Persistent Memory:** Stores conversations in a MySQL database.
## Preview

![2025-03-07T12_57_21 183Z](https://github.com/user-attachments/assets/ae78343e-c75a-403e-99f1-ff8289112082)



## Requirements
Ensure you have the following installed before running the chatbot:
- Python 3.9+
- MySQL Server
- Required Python libraries (install using `pip install -r requirements.txt`)

### Dependencies
```bash
pip install streamlit transformers sqlalchemy mysql-connector-python torch scikit-learn
```

## Setup Instructions
### 1. Database Configuration
Create a MySQL database and user for the chatbot:
```sql
CREATE DATABASE chatbot_db;
CREATE USER 'chatbot_user'@'localhost' IDENTIFIED BY '12345';
GRANT ALL PRIVILEGES ON chatbot_db.* TO 'chatbot_user'@'localhost';
FLUSH PRIVILEGES;
```

### 2. Running the Chatbot
Execute the following command to start the chatbot:
```bash
streamlit run app.py
```

## File Structure
```
.
├── chatbot.py           # Main chatbot application
├── requirements.txt     # List of dependencies
├── README.md            # Project documentation
```

## Usage
1. Open the chatbot interface in a browser.
2. Enter your message in the text box.
3. The chatbot will detect intent, generate a response, and store it in the database.
4. View past conversations in the chat history section.

## Acknowledgments
- **Hugging Face Transformers:** Used for Flan-T5 model.
- **Streamlit:** Interactive UI for chatbot.
- **SQLAlchemy:** ORM for database interactions.

## License
This project is licensed under the MIT License.

## Author
Abhishek Shrimali
