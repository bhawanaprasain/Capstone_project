from models.model import Query
from fastapi import APIRouter
from loguru import logger
from src.db_pipeline import chroma_db_handler
from src.inference import Chatbot


router = APIRouter(
    tags=["Chatbot"],
)

chatbot =Chatbot()

@router.post("/chat/")
async def chat(query: Query):
    try:
        question, history,client_name = (
            query.question,
            query.history,
            question.client_name
        )
        context = chroma_db_handler.query_collection(query_texts=[query], n_results=1)
        print(context)

        distance = context['distances'][0][0]
        response = chatbot.get_response(query, history, distance, context)
        return [{"message": response}]
    except Exception as e:
        logger.error(e)
        return [
            {
                "message": "I could not catch you. Can you please try rephrasing your query?"
            }
        ]
