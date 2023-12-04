from pydantic import BaseModel
from typing import List


class Query(BaseModel):
    history: List
    question: str
