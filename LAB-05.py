import streamlit as st
import requests
from openai import OpenAI
import json


def get_current_weather(location):
    API_key = st.secrets["open_weather_api_key"]
    if "," in location:
        location = location.split(",")[0].strip()

    urlbase = "https://api.openweathermap.org/data/2.5/"
    urlweather = f"weather?q={location}&appid={API_key}"
    url = urlbase + urlweather

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        temp = data['main']['temp'] - 273.15
        weather_description = data['weather'][0]['description']

        return json.dumps({
            "location": location,
            "temperature": round(temp, 2),
            "weather_description": weather_description
        })
    except Exception as e:
        return json.dumps({"error": f"Error fetching weather data: {str(e)}"})


client = OpenAI(api_key=st.secrets["openai_api_key"])


def get_weather_based_suggestions(user_input):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a witty weather expert who provides creative and concise weather-based suggestions. Always use Celsius for temperature. If no location is provided, use 'Syracuse NY' as the default."},
                {"role": "user", "content": user_input}
            ],
            functions=[
                {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                        },
                        "required": ["location"],
                    },
                }
            ],
            function_call="auto",
        )

        message = response.choices[0].message

        location = "Syracuse NY"  # Default location
        if hasattr(message, 'function_call') and message.function_call:
            function_name = message.function_call.name
            function_args = json.loads(message.function_call.arguments)
            if function_name == "get_current_weather":
                location = function_args.get("location", location)

        weather_data = get_current_weather(location)
        weather_dict = json.loads(weather_data)

        if "error" in weather_dict:
            return f"Oops! My weather crystal ball is a bit foggy for {location}. {weather_dict['error']}"

        second_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a witty weather expert who provides creative and concise weather-based suggestions. Use metaphors or similes to describe the weather. Be brief but engaging. Use the provided headings in your response."},
                {"role": "user", "content": user_input},
                {"role": "function", "name": "get_current_weather",
                    "content": weather_data},
                {"role": "user", "content": f"Describe {weather_dict['location']}'s weather. The temperature is {weather_dict['temperature']}°C, and the sky shows '{weather_dict['weather_description']}'. Provide:\n\n🌡️ Weather Report:\n[Describe the weather creatively in 1-2 sentences]\n\n👚 What to Wear:\n[Suggest appropriate clothing in 1 sentence]\n\n🌍 Overall Weather Vibe:\n[Summarize the day's feel in 1 short sentence]"}
            ],
        )
        return second_response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating suggestions: {str(e)}")
        return None


# Streamlit app
st.title("LAB-05 -- Weather-based Suggestion App")

# Input for user query
user_input = st.text_input(
    "Ask for weather-based suggestions (e.g., 'What should I wear today in New York? Is it good for a picnic?'):")

# Button to get suggestions
if st.button("Get Suggestions"):
    if user_input:
        suggestions = get_weather_based_suggestions(user_input)
        if suggestions:
            st.subheader("Weather-based Suggestions:")
            st.write(suggestions)
    else:
        st.warning("Please enter a query.")
