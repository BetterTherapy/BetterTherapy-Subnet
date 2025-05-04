from pydantic import BaseModel, Field
from typing import List, Dict
import bittensor as bt

class ResponseBreakdown(BaseModel):
    """Detailed breakdown of medicalresponse evaluation."""
    relevance: float = Field(description="Relevance to the user query (0-10)")
    accuracy: float = Field(description="Medical accuracy (0-10)")
    empathy: float = Field(description="Tone and empathy in response (0-10)")
    clarity: float = Field(description="Clarity of advice (0-10)")
    contextuality: float = Field(description="Use of chat history context (0-10)")

class TherapyResponseSchema(BaseModel):
    """Structured output schema for medical responses."""
    query: str = Field(description="User query")
    response_text: str = Field(description="medical response text")
    breakdown: ResponseBreakdown = Field(description="Breakdown of response evaluation")
    user_rating: float = Field(description="User rating (0-10)", default=1.0)

class TherapyInference:
    """Handles inference using the fine-tuned base model."""
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the fine-tuned base model (placeholder)."""
        # this load the latest fine-tuned model from miner-shared resources
        self.model = "medical_exam_model_v1"  # actual model loading logic
        bt.logging.info("Loaded fine-tuned base model")

    async def generate_therapy_response(self, query: str, chat_history: List[Dict[str, str]]) -> TherapyResponseSchema:
        """
        Generate a medical response using the fine-tuned model.
        """
        try:
            # Analyze chat history
            history_context = " ".join([msg["content"] for msg in chat_history]) if chat_history else "No prior context"
            
            # Simulate inference with the fine-tuned model
            response_text = f"Dr. AI: Response to '{query}' - Based on history ({history_context}), I recommend a medical session or medical review."

            # scores based on simulated analysis
            relevance = 8.5 if query.lower() in response_text.lower() else 6.0  # Check if query is addressed
            accuracy = 7.0 + (len(history_context.split()) * 0.1) if history_context != "No prior context" else 7.0  
            empathy = 8.0 if "recommend" in response_text.lower() or "session" in response_text.lower() else 5.0  # Empathetic tone
            clarity = 9.0 if len(response_text.split()) > 10 else 6.0  
            contextuality = 7.5 if history_context != "No prior context" else 4.0  # Use of chat history

            breakdown = ResponseBreakdown(
                relevance=relevance,
                accuracy=accuracy,
                empathy=empathy,
                clarity=clarity,
                contextuality=contextuality
            )

            return TherapyResponseSchema(
                query=query,
                response_text=response_text,
                breakdown=breakdown,
                user_rating=1.0  # Default, updated after user feedback
            )
        except Exception as e:
            bt.logging.error(f"Failed to generate medical response: {str(e)}")
            raise Exception(f"Failed to generate medical response: {str(e)}")

    async def update_user_rating(self, response: TherapyResponseSchema, user_rating: float):
        """Update the user rating for a response (step 6)."""
        response.user_rating = max(0, min(user_rating, 10))  # Ensure rating is between 0 and 10
        bt.logging.info(f"Updated user rating for query '{response.query}': {response.user_rating}")
        return response
