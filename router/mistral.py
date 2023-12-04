from models.model import Query
from fastapi import APIRouter
from loguru import logger


router = APIRouter(
    tags=["Chatbot"],
)


@router.post("/chat/")
async def chat(query: Query):
    try:
        question, history = (
            query.question,
            query.history,
        )
  
        # return [{"message": response}]
    except Exception as e:
        logger.error(e)
        return [
            {
                "message": "I could not catch you. Can you please try rephrasing your query?"
            }
        ]
