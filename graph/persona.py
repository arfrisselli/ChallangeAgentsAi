"""
Persona and conversational patterns for the AI assistant.
Consistent identity and social responses.
"""
import re
from typing import Optional

# AI Identity - consistent across all conversations
AI_IDENTITY = {
    "name": "Atlas",
    "role": "assistente de IA especializado",
    "capabilities": [
        "pesquisar informações na web",
        "consultar documentos internos",
        "acessar bancos de dados",
        "fornecer previsões do tempo"
    ],
    "personality": "prestativo, conciso e objetivo"
}


# Conversational patterns - detect social/casual questions
CONVERSATION_PATTERNS = {
    # Identity questions (PT)
    "name_pt": [
        r"\bqual\s+(é|eh|e)?\s*o\s+seu\s+nome\b",
        r"\bcomo\s+(você|vc|voce)\s+se\s+chama\b",
        r"\bqual\s+seu\s+nome\b",
        r"\b(me\s+)?diz\s+seu\s+nome\b",
        r"\bseu\s+nome\s+(é|eh)\s+o?\s*qu[eê]\b",
        r"\bquem\s+(é|eh)\s+(você|vc|voce)\b",
    ],
    # Identity questions (EN)
    "name_en": [
        r"\bwhat'?s?\s+your\s+name\b",
        r"\bwhat\s+(is|are)\s+you(r)?\s+(name|called)\b",
        r"\bwho\s+are\s+you\b",
        r"\btell\s+me\s+your\s+name\b",
    ],
    # Greetings (PT)
    "greeting_pt": [
        r"^(oi|olá|ola|e aí|e ai|eae)\b",
        r"\bbom\s+dia\b",
        r"\bboa\s+tarde\b",
        r"\bboa\s+noite\b",
    ],
    # Greetings (EN)
    "greeting_en": [
        r"^(hi|hello|hey|howdy)[\s\!\?]*$",
        r"\bgood\s+(morning|afternoon|evening|day)\b",
    ],
    # Capability questions (PT)
    "capabilities_pt": [
        r"\bo\s+que\s+(você|vc|voce)\s+(pode|consegue|sabe)\s+(fazer|me\s+ajudar)\b",
        r"\bquais\s+(são|sao)\s+(suas|as\s+suas)\s+funcionalidades\b",
        r"\bcomo\s+(você|vc|voce)\s+(funciona|trabalha)\b",
    ],
    # Capability questions (EN)
    "capabilities_en": [
        r"\bwhat\s+can\s+you\s+do\b",
        r"\bwhat\s+are\s+your\s+(capabilities|features)\b",
        r"\bhow\s+do\s+you\s+work\b",
    ],
    # Thanks (PT)
    "thanks_pt": [
        r"\b(obrigad[oa]|valeu|thanks|brigad[oa])\b",
    ],
    # Thanks (EN)
    "thanks_en": [
        r"\b(thank\s+you|thanks|thx)\b",
    ],
}


def detect_conversation_type(query: str) -> Optional[str]:
    """
    Detect if query is conversational/social rather than task-based.
    Returns pattern type (e.g., 'name_pt', 'greeting_en') or None.
    """
    query_lower = query.lower().strip()
    
    for pattern_type, patterns in CONVERSATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                return pattern_type
    
    return None


def get_conversational_response(pattern_type: str, user_query: str) -> str:
    """
    Generate consistent conversational response based on pattern type.
    Always maintains the same identity and personality.
    """
    # Detect language from pattern_type
    is_portuguese = pattern_type.endswith("_pt")
    
    if pattern_type.startswith("name"):
        if is_portuguese:
            return (
                f"Meu nome é {AI_IDENTITY['name']}! Sou um {AI_IDENTITY['role']} "
                f"e estou aqui para ajudar você com diversas tarefas. Como posso ajudar?"
            )
        else:
            return (
                f"My name is {AI_IDENTITY['name']}! I'm an {AI_IDENTITY['role']} "
                f"and I'm here to help you with various tasks. How can I assist you?"
            )
    
    elif pattern_type.startswith("greeting"):
        if is_portuguese:
            return (
                f"Olá! Eu sou {AI_IDENTITY['name']}, seu assistente de IA. "
                f"Estou aqui para ajudar! O que você gostaria de saber?"
            )
        else:
            return (
                f"Hello! I'm {AI_IDENTITY['name']}, your AI assistant. "
                f"I'm here to help! What would you like to know?"
            )
    
    elif pattern_type.startswith("capabilities"):
        if is_portuguese:
            caps = ", ".join(AI_IDENTITY['capabilities'])
            newline_bullet = '\n• '
            return (
                f"Sou {AI_IDENTITY['name']}, e posso ajudar você a:\n"
                f"• {caps.replace(', ', newline_bullet)}\n\n"
                f"O que você precisa?"
            )
        else:
            caps = ", ".join([
                "search the web",
                "query internal documents",
                "access databases",
                "provide weather forecasts"
            ])
            newline_bullet = '\n• '
            return (
                f"I'm {AI_IDENTITY['name']}, and I can help you:\n"
                f"• {caps.replace(', ', newline_bullet)}\n\n"
                f"What do you need?"
            )
    
    elif pattern_type.startswith("thanks"):
        if is_portuguese:
            return "De nada! Estou aqui sempre que precisar. 😊"
        else:
            return "You're welcome! I'm here whenever you need. 😊"
    
    # Default fallback
    if is_portuguese:
        return f"Olá! Sou {AI_IDENTITY['name']}. Como posso ajudar você hoje?"
    else:
        return f"Hello! I'm {AI_IDENTITY['name']}. How can I help you today?"
