# Gemma-3 OCR

---

Extract structured text from images using the powerful Gemma-3 Vision model! This Streamlit application provides a user-friendly interface to upload images and perform Optical Character Recognition (OCR), presenting the extracted text in a clear and organized Markdown format.

## Features

* **Image Upload:** Easily upload PNG, JPG, or JPEG image files.
* **Gemma-3 Vision Integration:** Leverages the `ollama` library to interact with the `gemma3:4b` model for advanced OCR capabilities.
* **Structured Output:** The extracted text is formatted into a readable Markdown structure, including headings, lists, or code blocks as necessary.
* **Clear Button:** A convenient "Clear" button to reset the results and upload a new image.
* **User-Friendly Interface:** Built with Streamlit for a simple and intuitive experience.

## Getting Started

Follow these steps to set up and run the Gemma-3 OCR application locally.

### Prerequisites

* **Python 3.x**
* **Ollama:** You'll need to have Ollama installed and the `gemma3:4b` model downloaded.
    * Download Ollama from [ollama.com](https://ollama.com/).
    * Run `ollama run gemma3:4b` in your terminal to download the model.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

    (Replace `<repository_url>` and `<repository_name>` with your actual repository details.)

2.  **Create a `requirements.txt` file** in the same directory as your Python script (`gemma3_ocr.py` or similar) with the following content:

    ```
    streamlit
    ollama
    Pillow
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the Streamlit application:**

    ```bash
    streamlit run your_app_name.py
    ```

    (Replace `your_app_name.py` with the actual name of your Python script, e.g., `gemma3_ocr.py`)

2.  The application will open in your web browser.

## Usage

1.  **Upload Image:** In the sidebar, click on "Choose an image..." to select a PNG, JPG, or JPEG file from your computer.
2.  **Extract Text:** Once the image is displayed in the sidebar, click the "**Extract Text 🔍**" button.
3.  **View Results:** The extracted text, formatted in Markdown, will appear in the main content area.
4.  **Clear Results:** Click the "**Clear 🗑️**" button in the top right to clear the current results and process a new image.

---

Made with ❤️ using Gemma-3 Vision Model
