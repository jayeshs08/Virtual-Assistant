from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3
import smtplib
import datetime
import os

# Load DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to generate response with DialoGPT
def generate_response(user_input):
    global chat_history_ids 
    
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

    # Generate response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Return the bot's response
    return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

# Function to speak out the response
def speak(response):
    engine.say(response)
    engine.runAndWait()

# Function to send email
def send_email(receiver_email, subject, message):
    # Configure SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('your_email@gmail.com', 'your_password')

    # Compose email
    email_content = f"Subject: {subject}\n\n{message}"

    # Send email
    server.sendmail('your_email@gmail.com', receiver_email, email_content)
    server.quit()

# Function to get current time
def get_current_time():
    current_time = datetime.datetime.now().strftime("%H:%M")
    return current_time

# Function to open applications
def open_application(application_name):
    os.system(f"start {application_name}")

# Initialize chat history
chat_history_ids = None

# Start conversation loop
while True:
    # Get user input
    user_input = input(">> User: ")

    # Check if user wants to end conversation
    if user_input.lower() == "end convo":
        print("Ending conversation...")
        speak("Goodbye!")
        break


    # Handle special commands
    if "send email" in user_input.lower():
        receiver_email = input("Enter receiver's email address: ")
        subject = input("Enter email subject: ")
        message = input("Enter email message: ")
        send_email(receiver_email, subject, message)

    elif "tell time" in user_input.lower():
        current_time = get_current_time()
        speak(f"The current time is {current_time}")

    elif "open application" in user_input.lower():
        application_name = input("Enter application name: ")
        open_application(application_name)

    # Generate response
    response = generate_response(user_input)

    # Speak out the response
    speak(response)

    # Print the response
    print("DialoGPT:", response)
