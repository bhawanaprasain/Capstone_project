# Capstone_project
Conversational AI using LLM

### Chat with PDF using ChromaDB and LLM

Welcome to our Capstone project! This project combines the power of ChromaDB and LLM (Large Language Model) to create a chat application that interacts user on health concerns. This README provides a guide on setting up the project, understanding its components, and getting started  with chat application.
## Features

FastAPI: Utilizes FastAPI, a modern, fast web framework for building APIs with Python 3.7+.
ChromaDB: Integrates ChromaDB for efficient storage and retrieval of chat conversations and PDF documents.
LLM (Large Language Model): Leverages LLM(Mistral 7b) to enhance the chat experience with advanced natural language understanding.
FAQ Integration: Allows users to get answers for questions that are specific to the servicer provider.

## Installation

# Clone the repository
```git clone https://github.com/bhawanaprasain/Capstone_project/```

# Change into the project directory:
```cd Capstone_project```

# Install dependencies:
```pip install -r requirements.txt```

# Change into the api directory:
```cd api```

## Usage
# Run the FastAPI application:
```uvicorn main:app --reload```

This starts the FastAPI development server.

Open your browser and go to ```http://localhost:8000```

# API Endpoints
```/chat```: Post messages and receive responses from the chatbot.
Request body
```
{
	"query":"What are the steps to prepare budget",
    "history" : [],
    "client_name" : "name_of_service_provider"
}
```


