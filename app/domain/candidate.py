from dataclasses import dataclass

@dataclass
class Candidate:
    """
    Represents a single candidate result from a model, including metadata such as latency, cost, and token usage.
    
    Attributes:
        model (str): The name of the model that generated the candidate.
        text (str): The generated text output.
        latency_s (float): Latency in seconds for generating the candidate.
        cost_usd (float): Cost in USD for generating the candidate.
        tokens_in (int): Number of input tokens used.
        tokens_out (int): Number of output tokens generated.
    """
    model: str
    text: str
    latency_s: float
    cost_usd: float
    tokens_in: int
    tokens_out: int
