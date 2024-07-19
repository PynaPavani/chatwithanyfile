# Chat with Any File Type

This project is a versatile chat application built using Streamlit, capable of handling various file types such as images, Excel files, CSV files, PDFs, and text files. The application leverages Ollama LangChain LLMs to generate responses, making it a powerful tool for querying and interacting with diverse data formats.

## Features

- **Multi-format File Handling**: Supports images, Excel (.xlsx), CSV, PDF, and text files.
- **Advanced Querying**: Uses Ollama LangChain LLMs to process and respond to user queries based on the content of the uploaded files.
- **Streamlit Integration**: A user-friendly interface powered by Streamlit, making the application accessible and easy to use.

## Getting Started

### Prerequisites

- Python 3.8 or later
- Streamlit
- LangChain
- Faiss
- Ollama LLMs
- Additional dependencies (specified in `requirements.txt`)

### Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/PynaPavani/Chat-with-Any-File-Type.git
   cd Chat-with-Any-File-Type
   ```

2. **Install Dependencies**

   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Application**

   ```sh
   streamlit run app.py
   ```

## Usage

1. Open the Streamlit app in your browser. You will see an interface to upload files.
2. Upload a file (image, Excel, CSV, PDF, or text file).
3. Enter your query related to the uploaded file.
4. The application will process your query using Ollama LangChain LLMs and provide a response.

## Project Structure

- `app.py`: The main Streamlit application script.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.

## Dependencies

Ensure you have the following Python packages installed (as listed in `requirements.txt`):

- streamlit
- langchain
- faiss
- ollama
