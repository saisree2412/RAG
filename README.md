# Hotel and Activities Finder

This application finds activities near a given location and then gives the nearby hotels to the activities using RAG approach, Gemini AI, and Google Maps API.

## Setup

### 1. Clone the repository

### 2. Create a virtual environment and install dependencies
On macOS and Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
On Windows:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

### 3. Set up environment variables
Create a .env file in the root of the project directory and add your API keys. You can use the .env.example file as a template

### 4. Run the application
Run the setup_ngrok.py script to start the Streamlit app and set up ngrok
```bash
python setup_ngrok.py
```

### 5. Access the application
After running the setup_ngrok.py script, you will see a URL printed in the console. Open this URL in your web browser to access the Streamlit app.





