from models.model import Query
from fastapi import APIRouter
from loguru import logger
from src.db import chroma_db_handler
from src.inference import Chatbot


router = APIRouter(
    tags=["Chatbot"],
)

chatbot =Chatbot(model_id="Bhawana/medicare")

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
        return [{"message": response}]
    except Exception as e:
        logger.error(e)
        return [
            {
                "message": "I could not catch you. Can you please try rephrasing your query?"
            }
        ]
