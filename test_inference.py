import requests
import os
import json

# --- Configuration ---
TEST_IMAGE_PATH = os.path.join('data', 'Straw6D_Raw', 'images', '0000.png')
API_URL = 'http://127.0.0.1:8000/api/analyze_image'

def run_test():
    """
    Sends a test image to the running API and prints the JSON response.
    """
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Error: Test image not found at '{TEST_IMAGE_PATH}'")
        return

    print(f"--- Running Inference Test ---")
    print(f"Image: {TEST_IMAGE_PATH}")
    print(f"API Endpoint: {API_URL}")

    try:
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': (os.path.basename(TEST_IMAGE_PATH), f, 'image/png')}
            response = requests.post(API_URL, files=files, timeout=30)

        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Pretty-print the JSON response
        response_json = response.json()
        print("\n--- API Response ---")
        if 'targets' in response_json:
            print(f"Found {len(response_json['targets'])} targets.")
        print(json.dumps(response_json, indent=2, ensure_ascii=False))

    except requests.exceptions.ConnectionError as e:
        print(f"\n--- Connection Error ---")
        print(f"Failed to connect to the API at {API_URL}.")
        print("Please make sure the backend server is running. You can start it with 'python app.py' or using the './run_project.ps1' script.")
    except requests.exceptions.RequestException as e:
        print(f"\n--- API Request Error ---")
        print(f"An error occurred while communicating with the API: {e}")
    except Exception as e:
        print(f"\n--- An unexpected error occurred ---")
        print(e)

if __name__ == '__main__':
    run_test()
