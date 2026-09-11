from langchain_huggingface import HuggingFaceEmbeddings
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    try:
        logger.info("Initializing HuggingFace Embeddings model")
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        logger.info("HuggingFace Embeddings model initialized successfully")
        _embedding_model = model
        return _embedding_model
    except Exception as e:
        error_message = CustomException("Error initializing HuggingFace Embeddings model: ", e)
        logger.error(str(error_message))
        raise error_message
    

