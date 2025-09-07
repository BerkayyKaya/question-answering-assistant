# Getting Started with IoT Data Question-Answering Assistant

This page will help you get started with question-answering assistant.
This project is a mini question-answering assistant that responds to natural language questions about IoT data such as temperature and humidity, collected from three different rooms in a house (living room, kitchen, bedroom). The assistant can also answer general knowledge questions.

# About the Project
The main purpose of the project is to find the most relevant data for user queries by performing a semantic search within unstructured text data. The found data is then used to generate a meaningful response.

When a user asks a specific question like "What was the temperature in the living room on 15 August 9 A.M.?", the model converts this question into vector representations. It then searches for the closest matches in a vector database. The retrieved results are provided as context to a language model, and a natural language answer is created for the user.

## Features
- Natural Language Querying: Ask questions about IoT data in natural language.

- Specific Data Access: Query temperature and humidity values for specific rooms (living room, kitchen, bedroom) and time periods.

- Semantic Search: Get fast and relevant results thanks to a vector database.

- General Knowledge Capability: The ability to answer general knowledge questions in addition to IoT data.

- Interactive Interface: A user-friendly demo interface developed with Streamlit.
Use the package manager

## Used Technologies
- **Python**

- **Streamlit:** For the demo showcase.

- **Hugging Face Transformers:** For language models and embeddings.

- **FAISS:** For the vector database.

- **LangChain:** To build the assistant logic and RAG integration.

## About Initializing Project
**To run this project on your local machine, follow the steps below.**

### 1. Clone the Repo
First clone the repository to your computer using following command:
```bash
git clone https://github.com/BerkayyKaya/question-answering-assistant.git
cd question-answering-assistant
```
### 2. Create a Virtual Environment 
To isolate the project libraries from your system's libraries, create a virtual environment.

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
Activate the virtual environment you just created.

- For Windows:
```bash
.\venv\Scripts\activate
```

- For macOS / Linux:
```bash
source venv/bin/activate
```
**If you did it right you should see (venv) at the beginning of your terminal prompt.**

### 4. Install the Required Libraries
Install all the Python libraries required to run the project from the requirements.txt file.

```bash
pip install -r requirements.txt
```

### 5. Start the Demo
Now you are ready to start the project demo. Run the following command in your terminal:

```bash
streamlit run ./streamlit/1_Giris.py
```
**This command will open the application in your default web browser.**

## Usage

When the application opens, you will see a chat interface. You can type both IoT data-related questions and general knowledge questions into this interface.

### Example Questions:
- IoT Data Queries:\
**What was the temperature in living room on 14 August 10 P.M.**\
**temperature in kitchen on 15 August**

- General Knowledge Queries:\
**What is the capital of Turkey?**\
**How can i reduce the temperature in living room?** 

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
