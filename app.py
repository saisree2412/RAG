import os
import requests
import streamlit as st
import google.generativeai as genai
from typing import List
import chromadb
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your API keys here
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_MAPS_API_KEY"] = os.getenv("GOOGLE_MAPS_API_KEY")
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

class GeminiEmbeddingFunction:
    def __call__(self, input):
        # Get GEMINI_API_KEY from environment variables
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        # Configure generative AI with the API key
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        # Embed content using the generative AI model
        response = genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)
        return response['embedding']

def create_or_load_chroma_db(documents, name="rag_maps"):
    # Create or load a ChromaDB collection
    chroma_client = chromadb.PersistentClient(path="/tmp/chromadb")
    try:
        db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        for i, d in enumerate(documents):
            db.add(documents=d, ids=str(i))
        total_documents = db.count()
        print(f"Total Indexed Documents: {total_documents}")
    except Exception as e:
        if "Collection" in str(e) and "already exists" in str(e):
            db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
            total_documents = db.count()
            print(f"Total Loaded Documents: {total_documents}")
        else:
            raise e
    return db

def get_coordinates(place_name):
    # Get the latitude and longitude for a given place name using Google Maps Geocoding API
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={place_name}&key={API_KEY}"
    response = requests.get(geocode_url)
    if response.status_code == 200:
        geocode_data = response.json()
        if geocode_data['results']:
            location = geocode_data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
    return None, None

def get_places_data(location, radius, place_type):
    # Get places data from Google Places API based on location, radius, and place type
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={location}&radius={radius}&type={place_type}&key={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('results', [])
    else:
        return []

def find_activities(location, radius=5000, type='tourist_attraction'):
    # Find tourist attractions near the specified location
    return get_places_data(location, radius, type)

def find_hotels_near_activities(activities, radius=1000):
    # Find hotels near the specified activities
    hotels = []
    for activity in activities:
        location = f"{activity['geometry']['location']['lat']},{activity['geometry']['location']['lng']}"
        hotels.extend(get_places_data(location, radius, 'lodging'))
    return hotels

def make_structured_prompt(query, context, sources):
    # Create a structured prompt for generative AI
    prompt = f"""
    Context: {context}.
    Query: {query}
    Sources: {sources}
    
    Task: Provide a structured response in JSON format containing:
    - A list of top 10 activities.
    - For each activity, a list of 5 nearby hotels.

    Format:
    [
        {{
            "activity": "Activity 1",
            "hotels": ["Hotel 1", "Hotel 2", "Hotel 3", "Hotel 4", "Hotel 5"]
        }},
        ...
    ]
    """
    print(f"Constructed Prompt: {prompt}")
    return prompt

def generate_answer(prompt):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    try:
        print(f"Sending prompt to the model: {prompt}")
        answer = model.generate_content(prompt)
        print(f"Raw response from model: {answer}")
        
        # Check if the response has 'candidates' and extract the text
        if hasattr(answer, 'candidates'):
            generated_text = answer.candidates[0].content.parts[0].text
            print(f"Generated Answer: {generated_text}")
            return generated_text
        else:
            raise ValueError("No candidates found in response")
    except Exception as e:
        print(f"Error generating content: {e}")
        return f"Error generating content: {e}"

def get_relevant_passages(query, db, n_results=3):
    # Get relevant passages from ChromaDB based on the query
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=gemini_api_key)
    model = "models/embedding-001"
    query_embedding = genai.embed_content(model=model, content=[query], task_type="retrieval_document")['embedding']
    total_documents = db.count()
    if n_results > total_documents:
        n_results = total_documents
    results = db.query(query_embeddings=query_embedding, n_results=n_results)['documents']
    flattened_results = [item for sublist in results for item in sublist]
    return flattened_results

def generate_response(db, query, context, sources):
    # Generate a response using the relevant passages, context, and sources
    relevant_passages = get_relevant_passages(query, db, n_results=3)
    if relevant_passages:
        prompt = make_structured_prompt(query, context, sources)
        answer = generate_answer(prompt)
        return answer
    else:
        return "No relevant information found."

def parse_model_response(response):
    # Parse the response generated by the model
    try:
        # Remove the "```json" and "```" parts if they are included in the response
        response = response.replace("```json", "").replace("```", "").replace("```JSON", "").strip()
        response = response.replace("\n", "").replace(" ", "")
        
        print(f"Cleaned model response: {response}")
        parsed_response = json.loads(response)
        st.write(parsed_response)
        return parsed_response
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []

def create_map_html(activities_and_hotels, API_KEY):
    # Create HTML for the map to display activities and hotels
    activity_markers = []

    for activity in activities_and_hotels:
        activity_lat, activity_lng = get_coordinates(activity["activity"])
        if activity_lat is not None and activity_lng is not None:
            marker = {
                "lat": activity_lat,
                "lng": activity_lng,
                "name": activity["activity"],
                "type": "activity",
                "highlighted": False,
                "hotels": activity["hotels"]
            }
            activity_markers.append(marker)

    if not activity_markers:
        st.write("No activities found to display on the map.")
        return ""

    map_html = f"""
    <html>
      <head>
        <title>Map</title>
        <script src="https://maps.googleapis.com/maps/api/js?key={API_KEY}&callback=initMap" async defer></script>
        <script type="text/javascript">
          var map;
          var markers = [];
          var activityMarkers = {json.dumps(activity_markers)};

          function initMap() {{
            map = new google.maps.Map(document.getElementById('map'), {{
              zoom: 12,
              center: {{lat: parseFloat(activityMarkers[0].lat), lng: parseFloat(activityMarkers[0].lng)}}
            }});

            for (var i = 0; i < activityMarkers.length; i++) {{
              var color = activityMarkers[i].highlighted ? 'red' : 'green';
              addMarker(activityMarkers[i], color);
            }}

            google.maps.event.addListener(map, 'click', function(event) {{
              clearMarkers();
              for (var i = 0; i < activityMarkers.length; i++) {{
                var color = activityMarkers[i].highlighted ? 'red' : 'green';
                addMarker(activityMarkers[i], color);
              }}
            }});
          }}

          function addMarker(location, color) {{
            var marker = new google.maps.Marker({{
              position: {{lat: parseFloat(location.lat), lng: parseFloat(location.lng)}},
              map: map,
              title: location.name,
              icon: 'http://maps.google.com/mapfiles/ms/icons/' + color + '-dot.png'
            }});
            markers.push(marker);

            if (location.type === 'activity') {{
              marker.addListener('click', function() {{
                showNearbyHotels(location);
              }});
            }} else if (location.type === 'hotel') {{
              marker.addListener('click', function() {{
                window.open(`https://www.google.com/search?q=${{encodeURIComponent(location.name)}}`);
              }});
            }}
          }}

          function showNearbyHotels(activity) {{
            clearMarkers();
            map.setCenter(new google.maps.LatLng(parseFloat(activity.lat), parseFloat(activity.lng)));
            for (var i = 0; i < activity.hotels.length; i++) {{
              get_coordinates(activity.hotels[i], function(hotelLat, hotelLng, hotelName) {{
                if (hotelLat && hotelLng) {{
                  var color = 'blue';
                  var hotelMarker = {{
                    lat: hotelLat,
                    lng: hotelLng,
                    name: hotelName,
                    type: 'hotel',
                    highlighted: false
                  }};
                  addMarker(hotelMarker, color);
                }}
              }});
            }}
          }}

          function clearMarkers() {{
            for (var i = 0; i < markers.length; i++) {{
              markers[i].setMap(null);
            }}
            markers = [];
          }}

          function get_coordinates(place_name, callback) {{
            var geocode_url = `https://maps.googleapis.com/maps/api/geocode/json?address=${{encodeURIComponent(place_name)}}&key={API_KEY}`;
            fetch(geocode_url)
              .then(response => response.json())
              .then(data => {{
                if (data.results && data.results.length > 0) {{
                  var location = data.results[0].geometry.location;
                  callback(location.lat, location.lng, place_name);
                }} else {{
                  callback(null, null, null);
                }}
              }})
              .catch(error => {{
                console.error('Error fetching coordinates:', error);
                callback(null, null, null);
              }});
          }}
        </script>
      </head>
      <body>
        <div id="map" style="width: 100%; height: 500px;"></div>
      </body>
    </html>
    """
    return map_html

# Streamlit app
st.title('Hotel and Activities Finder')

place_name = st.text_input('Enter a location (e.g., Salt Lake City, Utah):')

if st.button('Find Hotels and Activities'):
    if place_name:
        location_lat, location_lng = get_coordinates(place_name)
        if location_lat is not None and location_lng is not None:
            location = f"{location_lat},{location_lng}"
            activities = find_activities(location)
            hotels = find_hotels_near_activities(activities)

            st.header('Top Activities')
            st.json(activities)
            
            st.header('Nearby Hotels')
            st.json(hotels)

            documents = [str(activities), str(hotels)]
            db = create_or_load_chroma_db(documents)

            query = f"List top 10 activities and 5 hotels near each in {place_name}"
            context = "Finding activities and nearby hotels"
            sources = "Google Maps API"
            response = generate_response(db, query, context, sources)
            
            activities_and_hotels = parse_model_response(response)
            map_html = create_map_html(activities_and_hotels, API_KEY)
            st.components.v1.html(map_html, height=500)
        else:
            st.write("Failed to get coordinates for the place name.")
    else:
        st.write("Please enter a location.")
