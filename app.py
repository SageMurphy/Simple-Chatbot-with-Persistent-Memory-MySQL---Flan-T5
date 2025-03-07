import streamlit as st
from transformers import AutoTokenizer,  AutoModelForSequenceClassification, pipeline,AutoModelForSeq2SeqLM
from sqlalchemy import create_engine, Column, Integer, String, DateTime,Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch


st.title("Simple Chatbot with persistent memory (mysql)(Flan -T5)")


# database setup mysql

Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer , primary_key = True ,autoincrement= True)
    user_input = Column(Text, nullable = False)
    chatbot_response = Column(Text, nullable = False)
    timestamp = Column(DateTime, nullable = False)
# creating engine to connect to mysql 

engine = create_engine("mysql+mysqlconnector://chatbot_user:12345@localhost/chatbot_db")
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
    

# LOADING THE MODEL AND TOKENIZER
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model
    

tokenizer, model = load_model()

# intent detection setup
intents = {
     "greeting": ["hello", "hi", "hey", "good morning", "good afternoon"],
    "farewell": ["bye", "goodbye", "see you later", "farewell"],
    "general": ["how are you?", "tell me a joke", "what is the weather?","who am i?","do you recognize me?"],
    "about_me": ["what is your name?","tell me about yourself", "who are you","you?","Murphy?"],
    "search_web": ["search for", "find", "what is", "look up", "search the web for", "google", "find information about", "show me", "give me info on", "tell me about"]
}
all_texts = []
all_labels = []

for label, texts in intents.items():
    all_texts.extend(texts)
    all_labels.extend([label]*len(texts))

# vectorizer 

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(all_texts)


# classsifier

classifier = LogisticRegression()
classifier.fit(X, all_labels)

# intention detection function 
def detect_intent(user_input):
    user_input_vectorized = vectorizer.transform([user_input])
    intent = classifier.predict(user_input_vectorized)[0]
    return intent
# Few-shot examples
few_shot_examples = [
    "User: What is the weather like today?",
    "Chatbot: I'm sorry, I cannot provide real time information.",
    "User: Tell me a joke.",
    "Chatbot: Why don't scientists trust atoms? Because they make up everything!",
]

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

    # sentiment analysis setup
sentiment_pipeline = pipeline("sentiment-analysis")
def get_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result["label"], result["score"]

sentiment_threshold = 0.8


# sumarization setup
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()


# summarizer = pipeline("summarization")

def summarize_history(history):
    text = "\n".join(history)
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return summary

user_input = st.text_input("You:", key="user_input_1")

# transformer-based intent detection setup
intent_model_name = "distilbert-base-uncased" 
intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_name)
intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_name, num_labels=len(intents))

# Load the current model into the GPU if available.
# Load the current model into the GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
intent_model.to(device)  # Move the intent detection model to the correct device
model.to(device)  #  move the flan to same device 

def detect_intent_transformer(user_input):
    inputs = intent_tokenizer(user_input, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = intent_model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    return list(intents.keys())[predicted_class]


if user_input:
    session = Session()
    try:
        intent = detect_intent_transformer(user_input) #Use the transformer.
        # FOR SENTIMENT
        sentiment_label, sentiment_score = get_sentiment(user_input)
        if intent == "greeting":
            response = "Hello there! I'm Murphy, developed by Mr.Abhishek"
        elif intent == "farewell":
            response = "Goodbye!"
        elif intent == "about_me":
            response = "I'm Murphy, developed by Abhishek, i can learn by myself and i'm feeling good right now, seems like i am alive hehe!"
        # elif intent == "search_web":
        #     response = "Web search is disabled."
        else:
            
            # Check if history needs summarization
            if len(st.session_state.conversation_history) > 10:
                try:
                    summary = summarize_history(st.session_state.conversation_history)
                    st.session_state.conversation_history = [summary]
                except Exception as e:
                    st.write(f"Summarization failed: {e}")
            # construct few-shot prompt
            few_shot_context = "\n".join(few_shot_examples)
            # construct contextual prompt
            context = "\n".join(st.session_state.conversation_history)
            prompt = f"{few_shot_context}\n{context}\nUser: {user_input}\nChatbot:"
            inputs = tokenizer(user_input, return_tensors="pt").to(device)
            outputs = model.generate(** inputs, max_length = 50 , pad_token_id = tokenizer.eos_token_id)
            response = tokenizer.decode(outputs[0], skip_special_tokens = True)

# modifyinh responseds on the sentiment
        if sentiment_score > sentiment_threshold:
            if sentiment_label == "NEGATIVE":
                response = f"I sense you're feeling a bit down. {response}"
            elif sentiment_label == "POSITIVE":
                response = f"I'm glad you're in a good mood! {response}"

        # now i am storing conversations in the database
        conversation = Conversation( user_input = user_input, chatbot_response = response, timestamp = datetime.now())
        session.add(conversation)
        session.commit()


        # displaying the response
        st.write(f"You: {user_input}")
        st.write(f"Murphy:{response}")
  
# update the conversation history 

        st.session_state.conversation_history.append(f"User: {user_input}")
        st.session_state.conversation_history.append(f"Chatbot: {response}")

    except Exception as e:
        st.write(f"An error occurred: {e}")
        session.rollback()
    finally:
        session.close()


# dispaly the conversation from history 
session = Session()
try:
    conversations = session.query(Conversation).all()
    if conversations:
        st.subheader("Conversations History")
        for conv in conversations:
            st.write(f"You: {conv.user_input}")
            st.write(f"Murphy: {conv.chatbot_response}")
except Exception as e:
    st.write(f"Error retrieving conversations: {e}")
finally:
    session.close()