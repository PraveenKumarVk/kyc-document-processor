import requests
import json
import os
import re
import yaml
from fireworks.client import Fireworks
import fireworks.client
import base64
from PIL import Image
import pytesseract
from google.cloud import vision

fireworks.client.api_key = "fw_3ZGgq2M3JAQZYyeDDQ5QB65M"
client = Fireworks(api_key="fw_3ZGgq2M3JAQZYyeDDQ5QB65M")

def extract_text_with_google_vision(file_path):
    """
    Use Google Cloud Vision API to extract text from a document.
    """
    try:
        client = vision.ImageAnnotatorClient()

        with open(file_path, "rb") as image_file:
            content = image_file.read()

        # Perform text detection on the image
        image = vision.Image(content=content)
        response = client.text_detection(image=image)

        if response.error.message:
            raise Exception(response.error.message)

        return response.full_text_annotation.text
    
    except Exception as e:
        print(f"Error extracting text with Google Vision: {e}")
        return None
    
def extract_text_with_tesseract(file_path):
    """
    Use Tesseract as backup for offline situations.
    """
    try:
        image = Image.open(file_path)
        extracted_text = pytesseract.image_to_string(image)
        return extracted_text
    except Exception as e:
        print(f"Error extracting text with Tesseract: {e}")
        return None
    
def process_text_with_fireflow(extracted_text):
    """
    Use Fireworks Fireflow API to process the extracted text and convert it into meaningful structured data.
    """
    try:
        response = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": f"Extract the following information from the text: Name, Date of Birth, Document Number, Expiration Date, Address, and Document Type.\n "
                               f"Some documents can have name as FN(first name) and LN(lastname). Make sure to concatenate them if present.\n"
                               f"Make sure the labels in the response match with the labels I provided you with. "
                               f"For example, label 'name' has to be 'Name' but not first name or last name.\n\nText:\n{extracted_text}",
                }
            ],
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error processing text with Fireflow: {e}")
        return None

def extract_json_from_response(response_text):
    """
    Extract JSON from a response string using regex.

    Args:
        response_text (str): The raw response string containing JSON.

    Returns:
        dict: Parsed JSON object if extraction and parsing succeed.
    """
    try:
        # Use regex to find the JSON block within the response
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))  # Parse and return the JSON object
        else:
            print("No JSON block found in the response.")
            return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

def update_json_file(data, output_file):
    """
    Append new data to the output JSON file, ensuring no duplicates based on Document Number.
    """
    try:
        try:
            with open(output_file, "r") as json_file:
                existing_data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file is missing or empty, initialize an empty list
            existing_data = []

        # Ensure the existing data is a list
        if not isinstance(existing_data, list):
            print(f"Error: JSON file {output_file} does not contain a list.")
            return

        # Avoid duplicates by checking Document Number
        for entry in existing_data:
            if entry.get("Document Number") == data.get("Document Number"):
                print(f"Duplicate Document Number detected: {data.get('Document Number')}. Skipping entry.")
                return

        # Append the new data and save the file
        existing_data.append(data)
        with open(output_file, "w") as json_file:
            json.dump(existing_data, json_file, indent=4)
        print(f"Data added to {output_file}")
        
    except Exception as e:
        print(f"Error updating JSON file: {e}")

def save_to_json(extracted_text, output_file):
    """
    Use Fireflow to process extracted text and save the structured data to a JSON file.
    """
    try:
        # Use Fireflow API to structure the extracted text
        response = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-8b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": f"I'm providing you a structured text. I want you to format the text into JSON format and the output has to be just JSON-ready data. "
                               f"Date of Birth and Expiration Date fields must be in format MM-DD-YYYY. Document number should not exceed 10 characters. "
                               f"Address should not be split into multiple categories like state, city etc."
                               f"The value of Document type should be either 'Passport' or 'Drivers License'(case-sensitive). nothing else "
                               f"Text:\n{extracted_text}",
                }
            ],
        )

        # Parse the Fireflow response into JSON
        data = extract_json_from_response(response.choices[0].message.content)

        # Append the structured data to the output JSON file
        update_json_file(data=data, output_file=output_file)
        
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return None
