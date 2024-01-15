from models.model import Query
from fastapi import APIRouter
from loguru import logger
from src.db import chroma_db_handler
from src.inference import Chatbot
from src.postprocess import PostProcessing



router = APIRouter(
    tags=["Chatbot"],
)
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
model_id="Bhawana/medicare"
chatbot =Chatbot(model_id)
processor = PostProcessing()
@router.post("/chat/")
async def chat(query: Query):
    try:
        question, history,client_name = (
            query.question,
            query.history,
            query.client_name
        )

        context = chroma_db_handler.query_collection(query_texts=[question], n_results=1)
        print(context)

        distance = context['distances'][0][0]
        
        response = chatbot.get_response(question, history, distance, context)
        response = processor.postprocess_mistral_reponse(response)


        return [{"message": response}]
    except Exception as e:
        logger.error(e)
        return [
            {
                "message": "I could not catch you. Can you please try rephrasing your query?"
            }
        ]
