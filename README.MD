# KYC Document Processing

This project automates the extraction and processing of information from KYC (Know Your Customer) documents such as passports and driver's licenses. The pipeline utilizes OCR (Optical Character Recognition) and APIs to convert raw image data into structured JSON files.

## Features

- **Text Extraction**: Extracts text from images using:
  - Google Vision API
  - Tesseract OCR
- **Data Structuring**: Processes raw text using Fireflow (Fireworks AI) API to structure information like:
  - Name
  - Date of Birth
  - Document Number
  - Expiration Date
  - Address
  - Document Type
- **Duplicate Handling**: Ensures no duplicate entries are added to the output JSON file based on Document Number.
- **JSON Output**: Saves structured data in a standardized JSON format.

---

## Installation

### Prerequisites

- Python 3.8 or later
- Tesseract OCR installed on your system
  - On macOS: `brew install tesseract`
  - On Linux: `sudo apt install tesseract-ocr`
  - [Tesseract Installation Guide](https://github.com/tesseract-ocr/tesseract)

### Setup Instructions

1. Clone this repository:

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up the Google Cloud Vision credentials:
   - Follow the [Google Cloud Vision API setup guide](https://cloud.google.com/vision/docs/setup).
   - Export the `GOOGLE_APPLICATION_CREDENTIALS` environment variable:
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"  # macOS/Linux
     set GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\credentials.json"  # Windows
     ```

---

## Usage

### Input Files

- Place image files (e.g., `.jpg`, `.jpeg`, `.png`) in the `data/images` directory.

### Run the Script

To process all images and generate structured JSON output:

```bash
python main.py
```

### Output

- The structured data will be saved in the `data/output.json` file.

### Configuration

You can modify the following paths in `main.py`:

- **Input Directory**: Change `image_dir` to specify a different folder for input images.
- **Output File**: Change `output_file` to specify a different output JSON file.

---

## Project Structure

```
.
├── core
│   ├── kyc_processor.py    # Contains core functionality for text extraction and processing
├── data
│   ├── images              # Directory for input images
│   └── output.json         # Output JSON file
├── main.py                 # Main script to run the pipeline
├── requirements.txt        # List of required Python packages
└── README.md               # Project documentation
```

---

## Dependencies

- **fireworks-client**: Fireworks API client for Fireflow.
- **google-cloud-vision**: Google Vision API client for OCR.
- **pytesseract**: Python wrapper for Tesseract OCR.
- **Pillow**: Image processing library.
- **requests**: HTTP library for making API calls.
- **PyYAML**: For future YAML configuration handling (optional).

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## Limitations

- The Google Vision API requires an active internet connection and valid credentials.
- Tesseract OCR may produce less accurate results compared to Google Vision, depending on the quality of the input image.
- Fireflow API usage requires valid API credentials and sufficient quota.

---

## Design Choices and Tradeoffs

1. **Text Extraction**

- Choice: Tried both Google Vision API and Tesseract OCR for text extraction.
  - Reason: Google Vision API provides superior accuracy but requires an internet connection and valid credentials, making it less suitable for offline scenarios.
  - Tradeoff: Tesseract OCR is included as a fallback option for offline processing but may yield less accurate results for complex images.

2. **Data Structuring**

- Choice: Fireflow API is used to convert raw text into structured JSON.
  - Reason: Fireflow's capabilities allow for advanced text interpretation and formatting, reducing the need for custom parsing logic.
  - Tradeoff: Reliance on Fireflow introduces dependency on its API availability and sufficient quota.

3. **Duplicate Handling**

- Choice: Entries are identified and deduplicated based on Document Number.
  - Reason: The document number is a unique identifier that ensures no redundant entries are added.
  - Tradeoff: If Document Number is missing or inconsistent, duplicates might not be detected accurately.

4. **JSON Output Format**

- Choice: Standardized fields like Name, Date of Birth, and Document Type are enforced via Fireflow.
  - Reason: Ensures compatibility with downstream applications that rely on consistent field names.
  - Tradeoff: Some fields (e.g., Address) are consolidated for simplicity, which might limit detailed parsing.

---

## Future Enhancements

1. **Support for Additional Document Types**:

   - Extend the pipeline to process other identity documents like voter IDs, social security cards, or international IDs.

2. **Improved Error Handling**:

   - Implement robust error logging and reporting for failed extractions or API errors.
   - Include a retry mechanism for API calls to handle temporary failures.

3. **Data Validation and Standardization**:

   - Add validation for fields such as `Date of Birth`, `Document Number`, and `Expiration Date` to ensure consistency and format compliance.
   - Standardize field values (e.g., converting date formats to `YYYY-MM-DD`).

4. **Enhanced Address Parsing**:

   - Expand support for parsing and structuring multi-line addresses into fields like `Street`, `City`, `State`, and `Postal Code`.

5. **Batch Processing Support**:

   - Optimize the pipeline for processing large batches of documents in parallel for improved scalability.

6. **Integration with Third-Party Services**:

   - Integrate with external identity verification services to cross-check extracted information against official databases.

7. **Dashboard for Monitoring and Analytics**:

   - Develop a dashboard to visualize processed data, monitor document processing statuses, and identify trends in document submissions.

8. **Offline Mode**:

   - Enhance the Tesseract-based workflow for scenarios where Google Vision API is unavailable (e.g., restricted environments).

9. **Improved Duplicate Detection**:

   - Enhance duplicate detection by considering additional fields like `Name` and `Date of Birth` alongside `Document Number`.

10. **End-to-End Testing**:

    - Write unit and integration tests to ensure the pipeline works seamlessly for various document formats and edge cases.

11. **AI Model Fine-Tuning**:
    - Fine-tune Fireflow or integrate a custom model to improve extraction accuracy for region-specific document formats.

---

## Acknowledgments

- [Google Cloud Vision API](https://cloud.google.com/vision)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Fireworks AI](https://fireworks.ai/)
