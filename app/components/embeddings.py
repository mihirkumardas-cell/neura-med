from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
import os

logger = get_logger(__name__)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        logger.info("Initializing HuggingFace Inference API Embeddings model")
        api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
        model = HuggingFaceInferenceAPIEmbeddings(
            api_key=api_key,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        logger.info("HuggingFace Inference API Embeddings model initialized successfully")
        _embedding_model = model
        return _embedding_model
    except Exception as e:
        error_message = CustomException("Error initializing HuggingFace Embeddings model: ", e)
        logger.error(str(error_message))
        raise error_message
