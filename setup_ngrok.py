from pyngrok import ngrok
import subprocess
import os
from dotenv import load_dotenv
import sys

# Load environment variables from .env file
load_dotenv()

# Get NGROK_AUTH_TOKEN from environment variables
ngrok_auth_token = os.getenv("NGROK_AUTH_TOKEN")

# Set up ngrok with the authentication token
ngrok.set_auth_token(ngrok_auth_token)

# Get the full path to the Python executable
python_executable = sys.executable

# Run the Streamlit app in the background
subprocess.Popen([python_executable, "-m", "streamlit", "run", "app.py"])

# Start ngrok tunnel
public_url = ngrok.connect(8501)
print(f"Streamlit app is live at: {public_url}")
