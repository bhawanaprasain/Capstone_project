import re
import spacy
from loguru import logger

class PostProcessing:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def remove_sentences_with_url(text, pattern):
        """
        Removes sentences from the input text that contain a URL based on the provided pattern.

        Args:
            text (str): The input text from which sentences will be removed.
            pattern (str): The regular expression pattern to match URLs.

        Returns:
            str: The modified text with sentences containing URLs removed.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        remaining_sentences = [
            sentence for sentence in sentences if not re.search(pattern, sentence)
        ]
        new_text = " ".join(remaining_sentences)
        return new_text

    @staticmethod
    def clean_hyphens(text):
        """
        Cleans the response text by removing specific patterns and sentences containing URLs.

        Args:
            text (str): The response text to be cleaned.

        Returns:
            str: The cleaned response text.
        """
        if text.startswith("-"):
            text = text[1:]
            return text
        else:
            return text

    @staticmethod
    def remove_incomplete_sentence(text):
        """
        Cleans the response text by removing specific patterns and sentences containing URLs.

        Args:
            text (str): The response text to be cleaned.

        Returns:
            str: The cleaned response text.
        """
        if not (text.endswith(".") or text.endswith("!") or text.endswith("?")) and (
            text.count(".") > 1
        ):
            text = text[1:]
            # Find the last index of '?', '.', or '!'
            last_punctuation_index = max(
                text.rfind("?"), text.rfind("."), text.rfind("!")
            )
            text = text[: last_punctuation_index + 1]
            return text
        else:
            return text

    @staticmethod
    def remove_prompt(text):
        text = text.replace(
            "Now answer this question from human as an AI assistant.", ""
        ).replace("Do not repeat what human says.", "")
        return text

    @staticmethod
    def find_end_of_response(response):
        pattern = r"<([^>]+)>"
        matches = list(re.finditer(pattern, response))
        if matches:
            logger.info(matches)
            for match in matches[:1]:
                start_index = match.start()
                ai_response = response[:start_index]
                return ai_response
        else:
            return response

    def postprocess_mistral_reponse(self,response):
        logger.info(response)
        matches = [match.start() for match in re.finditer(r"\[/INST\]", response)]
        if matches:
            response_start_index = matches[-1]  # Get the last occurrence
            response = response[response_start_index + 7 :]  # Get the last occurrence]
        response =  self.find_complete_sentences(response)
        return response


    def is_complete_sentence(self, sentence):
        # Check if the sentence ends with ".", "!", or "?"
        return True if sentence[-1] in "?.!" else False
    
    def filter_followup_questions(self,sentences):
        followup_queries = [(index, sentence) for index, sentence in enumerate(sentences) if sentence.endswith('?')]
        if followup_queries:
            index_of_first_query = followup_queries[0][0]
            if followup_queries:
                sentences = sentences[:index_of_first_query+1]
        return sentences
    
    def find_complete_sentences(self,response):
        try:
            response = response.replace('</s>','')
            doc = self.nlp(response)
            # logger.info(response)
            sentences = [sent.text.strip() for sent in doc.sents ]
            logger.info(sentences)
            logger.info("\n")
           

            # Check if the last sentence is complete
            is_last_sentence_complete = self.is_complete_sentence(sentences[-1]) if sentences else False
            if not is_last_sentence_complete or len(sentences[-1])<3:
                sentences = sentences[:-1]
            sentences = self.filter_followup_questions(sentences)
            sentences = self.remove_incomplete_list_from_response(sentences)
            logger.info(sentences)
            return ' '.join(sentences) 
        except Exception as e:
            logger.info(response)
            logger.error(e)
    
    def remove_incomplete_list_from_response(self,sentences):
        pattern = r'[0-9]+\.|([ivxlc]+)\.'
        last_sentence = sentences[-1]
        match = re.search(pattern, last_sentence[-3:])
        if match:
            sentences[-1] = re.sub(pattern, '', sentences[-1])
        return sentences




