# AI Document Query Bot

A Slack bot that processes PDF documents and answers user queries based on the content of the document using OpenAI's language models and a FAISS vector database.

![AI Document Query Bot](screenshots/bot_screenshot.png)  <!-- Add a screenshot of the bot in action -->

## Features

- **Upload PDFs**: Seamlessly upload PDF files through Slack.
- **Text Extraction**: Extract text from PDFs and store it in a FAISS vector database.
- **Question Answering**: Use OpenAI's language model to answer questions based on the document content.
- **Error Handling**: Graceful error handling and informative messages.

## Technologies Used

- **Slack Bolt**: For interacting with Slack API.
- **OpenAI**: For natural language processing and generating responses.
- **FAISS**: For efficient similarity search and text indexing.
- **Transformers**: For encoding text using transformer models.
- **PyPDF2**: For extracting text from PDF files.
- **Python**: For the implementation.

## Installation

Follow these steps to set up and run the project:

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/ai-document-query-bot.git
