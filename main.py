# filename: multi_waypoint_api.py
import json
import re
import base64
import polyline
import requests
import openai
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, List, Optional
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env





# -----------------------------------------------------------
# 1. Your Credentials and Setup
# -----------------------------------------------------------
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")

openai.api_key = OPENAI_API_KEY

# -----------------------------------------------------------
# 2. FastAPI Schema Definitions
# -----------------------------------------------------------
class WaypointDetail(BaseModel):
    name: str
    address: str
    coordinates: List[float]  # [lat, lng]
    type: str
    hours: List[str] = []
    photos: List[str] = []

class LegInfo(BaseModel):
    distance: str
    duration: str

class DirectionsResponse(BaseModel):
    polyline: Optional[str]
    instructions: List[str] = []
    waypoints: List[WaypointDetail] = []
    round_trip: bool = False
    notes: str = ""
    legs: List[LegInfo] = []


class PromptRequest(BaseModel):
    prompt: str

# -----------------------------------------------------------
# 3. System Prompt for the LLM
# -----------------------------------------------------------
SYSTEM_PROMPT = """
You are a helpful assistant that extracts a sequence of waypoints from a user's navigation request.
The user may specify multiple stops (e.g. "Go from X to Y, then to Z"), 
and may or may not ask for a round trip.

However, sometimes the user only provides a general area or request like "plan me a date in Scarborough" or
"Where should I visit in New York on the weekend?" or "Tour me around downtown".

When the user request is this broad:
1. Assume the user wants 2-4 interesting or relevant stops (like a coffee shop, a scenic point, a restaurant, etc.).
2. Use the provided area or city as the first waypoint (treat it as an 'address'), unless the user explicitly provides an origin elsewhere.
3. Then add a few 'place_type' waypoints relevant to the request (e.g., "coffee shop", "restaurant", "scenic lookout").
4. Make sure to preserve the JSON structure and output valid JSON only.

You will output valid JSON with the following structure:

{
  "waypoints": [
    {
      "type": "address" | "place_type",
      "value": "string describing the place"
    },
    ...
  ],
  "round_trip": true|false,
  "extra_notes": "any clarifications or ambiguities"
}

Rules/notes:
- waypoints should appear in the order they are mentioned or, if the user is vague, propose a logical order.
- if the user wants "nearest coffee shop," use "type": "place_type" and "value": "coffee shop".
- if the user references a specific location like "UTSC" or "Yorkdale Mall," use "type": "address" and "value": that location.
- If the user says "home," treat it as an address (unless uncertain).
- If the user wants to return to the original starting point, set "round_trip": true.
- ONLY return valid JSON. No extra commentary.
- If unsure how to parse the user input, add clarifications in "extra_notes".
"""


# -----------------------------------------------------------
# 4. LLM Parsing for Waypoints
# -----------------------------------------------------------
def extract_waypoints(user_prompt: str) -> dict:
    """
    Ask the LLM (GPT-3.5-turbo) to parse the user's prompt into structured JSON 
    with multiple waypoints and round_trip info.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        raw_json = response.choices[0].message.content
        parsed = json.loads(raw_json)
        print(parsed)
        return parsed
    except Exception as e:
        print("Error extracting waypoints:", e)
        # Return a default structure if parsing fails
        return {
            "waypoints": [],
            "round_trip": False,
            "extra_notes": "Error or invalid JSON from LLM."
        }

# -----------------------------------------------------------
# 5. Geocoding and Places
# -----------------------------------------------------------
def geocode_address(address: str):
    """
    Use Google Maps Geocoding API to turn an address string into lat/lng.
    Returns (lat, lng) or None if not found.
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": GOOGLE_MAPS_API_KEY
    }
    resp = requests.get(url, params=params).json()
    if resp["status"] == "OK":
        location = resp["results"][0]["geometry"]["location"]
        return (location["lat"], location["lng"])
    else:
        return None

def find_nearest_place(origin_lat: float, origin_lng: float, place_type: str):
    """
    Use Google Places API to find the nearest place matching `place_type`.
    Returns (lat, lng, name, address, place_id) or None if not found.
    """
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": GOOGLE_MAPS_API_KEY,
        "location": f"{origin_lat},{origin_lng}",
        "rankby": "distance",
        "keyword": place_type
    }
    resp = requests.get(url, params=params).json()
    if resp["status"] == "OK" and len(resp["results"]) > 0:
        place = resp["results"][0]
        lat = place["geometry"]["location"]["lat"]
        lng = place["geometry"]["location"]["lng"]
        name = place["name"]
        address = place.get("vicinity", "")
        place_id = place.get("place_id", "")
        return (lat, lng, name, address, place_id)
    return None

def get_place_details(place_id: str):
    """
    Use Google Places Details API to fetch hours and photos for a given place_id.
    """
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "key": GOOGLE_MAPS_API_KEY,
        "place_id": place_id,
        "fields": "name,formatted_address,geometry,opening_hours,photos"
    }
    resp = requests.get(url, params=params).json()
    if resp["status"] == "OK":
        result = resp["result"]
        hours = result.get("opening_hours", {}).get("weekday_text", [])
        photos = result.get("photos", [])
        photo_refs = [
            f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photo_reference={p['photo_reference']}&key={GOOGLE_MAPS_API_KEY}"
            for p in photos
        ]
        return {"hours": hours, "photos": photo_refs}
    return {"hours": [], "photos": []}

# -----------------------------------------------------------
# 6. Directions with Multiple Waypoints
# -----------------------------------------------------------
def get_directions_with_waypoints(waypoint_coords_list: List[tuple]):
    if len(waypoint_coords_list) < 2:
        return None, [], []

    origin = f"{waypoint_coords_list[0][0]},{waypoint_coords_list[0][1]}"
    destination = f"{waypoint_coords_list[-1][0]},{waypoint_coords_list[-1][1]}"

    waypoints_param = "|".join(
        [f"{lat},{lng}" for (lat, lng) in waypoint_coords_list[1:-1]]
    ) if len(waypoint_coords_list) > 2 else None

    directions_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": destination,
        "key": GOOGLE_MAPS_API_KEY
    }
    if waypoints_param:
        params["waypoints"] = waypoints_param

    resp = requests.get(directions_url, params=params).json()
    if resp["status"] == "OK":
        route = resp["routes"][0]
        overview_poly = route["overview_polyline"]["points"]

        instructions = []
        legs_metadata = []

        for leg in route["legs"]:
            leg_distance = leg["distance"]["text"]
            leg_duration = leg["duration"]["text"]
            legs_metadata.append({
                "distance": leg_distance,
                "duration": leg_duration
            })

            for step in leg["steps"]:
                step_text = re.sub('<[^<]+?>', '', step["html_instructions"])
                instructions.append(f"{step_text} ({step['distance']['text']})")

        return overview_poly, instructions, legs_metadata
    else:
        return None, [], []


# -----------------------------------------------------------
# 7. Main pipeline: multiple waypoints
# -----------------------------------------------------------
def process_user_prompt(user_prompt: str):
    """
    End-to-end pipeline:
      1) Extract array of waypoints + round_trip from LLM
      2) Convert each waypoint to lat/lng 
         - if address => geocode
         - if place_type => search near the previous waypoint
      3) Build a single route with all stops in order
      4) If round_trip=True, append the first stop again
      5) Return the polyline, step instructions, waypoint details, round_trip, and notes
    """
    # 1) Get structured waypoints from LLM
    parsed = extract_waypoints(user_prompt)
    waypoints = parsed.get("waypoints", [])
    round_trip = parsed.get("round_trip", False)
    notes = parsed.get("extra_notes", "")

    if not waypoints:
        return {
            "polyline": None,
            "instructions": [],
            "waypoints": [],
            "round_trip": False,
            "notes": "I couldn't parse any valid stops from your request."
        }

    coords_list = []
    detailed_waypoints = []

    # 2) Convert each waypoint to lat/lng and collect details
    for i, wp in enumerate(waypoints):
        wp_type = wp["type"]
        wp_value = wp["value"]
        place_details = {"hours": [], "photos": []}

        # Error if the first waypoint is a "place_type" with no origin
        if i == 0 and wp_type == "place_type":
            return {
                "polyline": None,
                "instructions": [],
                "waypoints": [],
                "round_trip": False,
                "notes": "Your first waypoint is a place type, but no origin was provided."
            }

        if wp_type == "address":
            loc = geocode_address(wp_value)
            if loc is None:
                return {
                    "polyline": None,
                    "instructions": [],
                    "waypoints": [],
                    "round_trip": round_trip,
                    "notes": f"Could not geocode address: {wp_value}"
                }
            # Attempt to fetch extended place details (optional)
            place_info = find_nearest_place(loc[0], loc[1], wp_value)
            if place_info and len(place_info) == 5:
                _, _, _, _, place_id = place_info
                place_details = get_place_details(place_id)

            coords_list.append(loc)
            detailed_waypoints.append({
                "name": wp_value,
                "address": wp_value,
                "coordinates": loc,
                "type": "Address",
                "hours": place_details["hours"],
                "photos": place_details["photos"]
            })

        elif wp_type == "place_type":
            if not coords_list:
                return {
                    "polyline": None,
                    "instructions": [],
                    "waypoints": [],
                    "round_trip": round_trip,
                    "notes": "No origin available for place search. Please specify an address first."
                }
            origin_lat, origin_lng = coords_list[-1]
            place_info = find_nearest_place(origin_lat, origin_lng, wp_value)
            if place_info is None:
                return {
                    "polyline": None,
                    "instructions": [],
                    "waypoints": [],
                    "round_trip": round_trip,
                    "notes": f"Could not find a nearby place for: {wp_value}"
                }
            lat, lng, name, addr, place_id = place_info
            place_details = get_place_details(place_id)

            coords_list.append((lat, lng))
            detailed_waypoints.append({
                "name": name,
                "address": addr,
                "coordinates": (lat, lng),
                "type": f"Place Type - {wp_value}",
                "hours": place_details["hours"],
                "photos": place_details["photos"]
            })

    # 3) Handle round trip
    if round_trip and len(coords_list) > 0:
        coords_list.append(coords_list[0])
        detailed_waypoints.append(detailed_waypoints[0])  # duplicate

    if len(coords_list) < 2:
        return {
            "polyline": None,
            "instructions": [],
            "waypoints": [],
            "round_trip": round_trip,
            "notes": "Not enough waypoints to create a route."
        }

    # 4) Request directions from Google
    overall_poly, step_instructions, leg_metadata = get_directions_with_waypoints(coords_list)
    if not overall_poly:
        return {
            "polyline": None,
            "instructions": [],
            "waypoints": detailed_waypoints,
            "round_trip": round_trip,
            "notes": "Directions request failed. Please try again."
        }

    # 5) Return everything in a structured dict
    return {
        "polyline": overall_poly,
        "instructions": step_instructions,
        "waypoints": detailed_waypoints,
        "round_trip": round_trip,
        "notes": notes,
        "legs": leg_metadata

    }

# -----------------------------------------------------------
# 8. FastAPI App
# -----------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing. Replace with specific domains in production.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


@app.post("/api/directions", response_model=DirectionsResponse)
def get_directions(request: PromptRequest):
    """
    POST /api/directions
    Body: { "prompt": "Your navigation request" }

    Returns JSON with:
      polyline: Google Maps overview polyline string
      instructions: textual turn-by-turn instructions
      waypoints: list of waypoint details
      round_trip: boolean
      notes: any extra clarifications from the parser
    """
    result = process_user_prompt(request.prompt)
    return DirectionsResponse(
        polyline=result["polyline"],
        instructions=result["instructions"],
        waypoints=[
            WaypointDetail(
                name=wp["name"],
                address=wp["address"],
                coordinates=list(wp["coordinates"]),
                type=wp["type"],
                hours=wp.get("hours", []),
                photos=wp.get("photos", [])
            )
            for wp in result["waypoints"]
        ],
        round_trip=result["round_trip"],
        notes=result["notes"],
        legs=result.get("legs", [])
    )


# For local testing (e.g., `python multi_waypoint_api.py`)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
