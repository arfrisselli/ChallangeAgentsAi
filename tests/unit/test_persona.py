"""
Unit tests for persona/conversational system.
Tests identity consistency and pattern detection.
"""
import pytest
from graph.persona import (
    AI_IDENTITY,
    detect_conversation_type,
    get_conversational_response,
)


class TestPersonaIdentity:
    """Test that AI identity is consistent."""
    
    def test_ai_has_consistent_name(self):
        """AI must always have the same name."""
        assert AI_IDENTITY["name"] == "Atlas"
    
    def test_ai_has_role(self):
        """AI must have a defined role."""
        assert "role" in AI_IDENTITY
        assert len(AI_IDENTITY["role"]) > 0
    
    def test_ai_has_capabilities(self):
        """AI must have defined capabilities."""
        assert "capabilities" in AI_IDENTITY
        assert len(AI_IDENTITY["capabilities"]) >= 4


class TestConversationDetection:
    """Test conversation pattern detection."""
    
    def test_detect_name_question_pt(self):
        """Detect Portuguese name questions."""
        queries = [
            "Qual é o seu nome?",
            "Como você se chama?",
            "Qual seu nome?",
            "Me diz seu nome",
        ]
        for query in queries:
            result = detect_conversation_type(query)
            assert result == "name_pt", f"Failed for: {query}"
    
    def test_detect_name_question_en(self):
        """Detect English name questions."""
        queries = [
            "What's your name?",
            "What is your name?",
            "Who are you?",
            "Tell me your name",
        ]
        for query in queries:
            result = detect_conversation_type(query)
            assert result == "name_en", f"Failed for: {query}"
    
    def test_detect_greeting_pt(self):
        """Detect Portuguese greetings."""
        queries = ["Oi", "Olá", "Bom dia", "Boa tarde", "Boa noite"]
        for query in queries:
            result = detect_conversation_type(query)
            assert result == "greeting_pt", f"Failed for: {query}"
    
    def test_detect_greeting_en(self):
        """Detect English greetings."""
        queries = ["Hi", "Hello", "Hey", "Good morning", "Good afternoon"]
        for query in queries:
            result = detect_conversation_type(query)
            assert result == "greeting_en", f"Failed for: {query}"
    
    def test_detect_capabilities_pt(self):
        """Detect Portuguese capability questions."""
        queries = [
            "O que você pode fazer?",
            "Quais são suas funcionalidades?",
            "Como você funciona?",
        ]
        for query in queries:
            result = detect_conversation_type(query)
            assert result == "capabilities_pt", f"Failed for: {query}"
    
    def test_detect_thanks_pt(self):
        """Detect Portuguese thanks."""
        queries = ["Obrigado", "Obrigada", "Valeu", "Thanks", "Brigado"]
        for query in queries:
            result = detect_conversation_type(query)
            assert result == "thanks_pt", f"Failed for: {query}"
    
    def test_no_detection_for_task_queries(self):
        """Task queries should not be detected as conversational."""
        queries = [
            "Qual o clima em São Paulo?",
            "What's the weather in London?",
            "Liste os produtos do banco de dados",
            "Search for Python tutorials",
        ]
        for query in queries:
            result = detect_conversation_type(query)
            assert result is None, f"False positive for: {query}"


class TestConversationalResponses:
    """Test conversational response generation."""
    
    def test_name_response_pt_contains_identity(self):
        """Portuguese name response must contain AI name."""
        response = get_conversational_response("name_pt", "Qual é o seu nome?")
        assert AI_IDENTITY["name"] in response
        assert "Meu nome é" in response or "Me chamo" in response
    
    def test_name_response_en_contains_identity(self):
        """English name response must contain AI name."""
        response = get_conversational_response("name_en", "What's your name?")
        assert AI_IDENTITY["name"] in response
        assert "My name is" in response or "I'm" in response
    
    def test_name_response_is_always_same(self):
        """Name responses must be consistent across calls."""
        response1 = get_conversational_response("name_pt", "Qual seu nome?")
        response2 = get_conversational_response("name_pt", "Como você se chama?")
        # Both should contain the same name
        assert AI_IDENTITY["name"] in response1
        assert AI_IDENTITY["name"] in response2
    
    def test_greeting_response_pt_is_friendly(self):
        """Portuguese greeting should be friendly."""
        response = get_conversational_response("greeting_pt", "Oi")
        assert any(word in response for word in ["Olá", "Oi", AI_IDENTITY["name"]])
    
    def test_greeting_response_en_is_friendly(self):
        """English greeting should be friendly."""
        response = get_conversational_response("greeting_en", "Hello")
        assert any(word in response for word in ["Hello", "Hi", AI_IDENTITY["name"]])
    
    def test_capabilities_response_pt_lists_features(self):
        """Portuguese capabilities response should list features."""
        response = get_conversational_response("capabilities_pt", "O que você pode fazer?")
        # Should mention at least some capabilities
        assert any(cap in response.lower() for cap in ["pesquisar", "clima", "banco", "documentos"])
    
    def test_thanks_response_pt_is_polite(self):
        """Portuguese thanks response should be polite."""
        response = get_conversational_response("thanks_pt", "Obrigado")
        assert any(word in response for word in ["nada", "sempre", "precisar"])
    
    def test_response_matches_query_language(self):
        """Response language should match query language."""
        # Portuguese query -> Portuguese response
        response_pt = get_conversational_response("name_pt", "Qual seu nome?")
        assert any(word in response_pt for word in ["Meu", "Sou", "Estou"])
        
        # English query -> English response
        response_en = get_conversational_response("name_en", "What's your name?")
        assert any(word in response_en for word in ["My", "I'm", "I am"])
