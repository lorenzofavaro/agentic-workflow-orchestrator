from enum import Enum

class Skill(Enum):
    """
    Enum representing various skills or capabilities that a model or agent can possess.
    
    Members:
        ANALYZE: Analytical skills.
        REASON: Reasoning abilities.
        SUMMARIZE: Summarization skills.
        CODE: Coding or programming skills.
        MATH: Mathematical skills.
        PLAN: Planning or strategy skills.
    """
    ANALYZE = "analyze"
    REASON = "reason"
    SUMMARIZE = "summarize"
    CODE = "code"
    MATH = "math"
    PLAN = "plan"