import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.kyc_processor import extract_text_with_google_vision, process_text_with_fireflow, save_to_json

def main():
    image_dir = "data/images"
    output_file = "data/output.json"

    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(image_dir, filename)
            print(f"Processing file: {file_path}")

            # Extraction of text from the image using Google Vision API
            extracted_text = extract_text_with_google_vision(file_path)
            if not extracted_text:
                print(f"Failed to extract text from {file_path}.")
                continue

            # Processing the extracted text using Fireflow to structure it into a meaningful format
            structured_data = process_text_with_fireflow(extracted_text)
            if not structured_data:
                print(f"Failed to process text for {file_path}.")
                continue
            
            # Saving the structured data into the output JSON file
            save_to_json(structured_data, output_file)

if __name__ == "__main__":
    main()
