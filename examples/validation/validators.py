from typing import Any, Dict
from guardrails import Validator
from guardrails.classes import FailResult, PassResult, ValidationResult


TOXIC_WORDS = ["asshole", "damn"]

class BasicToxicLanguage(Validator):
    data_type = "string"
    
    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        is_toxic_language = any(toxic_word in value for toxic_word in TOXIC_WORDS)

        # if a value contains toxic words we return FailResult otherwise PassResult
        if is_toxic_language:
            for toxic_word in TOXIC_WORDS:
                value = value.replace(toxic_word, "")
            return FailResult(
                error_message=f"Value '{value}' contains toxic language including words: {TOXIC_WORDS} which is not allowed.",
                fix_value=value,
            )

        return PassResult()