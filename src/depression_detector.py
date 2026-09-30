"""
Depression detection and BDI-II scoring system.
"""

import re
import warnings
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Reduce noise when loading sentence-transformers (BertModel report, deprecation)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


@dataclass
class DepressionPrediction:
    """Depression prediction result."""
    bdi_score: int
    key_symptoms: List[str]
    confidence: float
    turn_count: int


class DepressionDetector:
    """Detect depression from conversations using SLM-based analysis."""
    
    def __init__(self, config: dict):
        """
        Initialize depression detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.bdi_symptoms = config['bdi_symptoms']
        self.max_symptoms = config['detection']['max_symptoms']
        self.symptom_threshold = config['detection']['symptom_threshold']
        self.early_stop_threshold = config['conversation']['early_stop_threshold']
        
        # Load sentence transformer for semantic analysis (SLM: 22M parameters)
        with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define symptom keywords and patterns
        self.symptom_patterns = self._build_symptom_patterns()
        
        # Define depression indicators
        self.depression_indicators = [
            "sad", "depressed", "hopeless", "worthless", "empty",
            "tired", "fatigue", "exhausted", "no energy",
            "sleep", "insomnia", "oversleeping", "can't sleep",
            "appetite", "eating", "not hungry", "overeating",
            "concentrate", "focus", "difficulty thinking",
            "guilt", "guilty", "blame myself",
            "suicide", "suicidal", "end it all", "not worth living",
            "interest", "pleasure", "enjoy", "nothing matters",
            "crying", "tears", "cry",
            "irritable", "angry", "annoyed",
            "indecisive", "can't decide", "uncertain"
        ]
    
    def _build_symptom_patterns(self) -> Dict[str, List[str]]:
        """Build keyword patterns for each BDI-II symptom."""
        patterns = {
            "Sadness": ["sad", "depressed", "down", "unhappy", "melancholy", "blue"],
            "Pessimism": ["hopeless", "pessimistic", "no future", "nothing will improve"],
            "Past Failure": ["failure", "failed", "disappointed", "let down"],
            "Loss of Pleasure": ["no pleasure", "can't enjoy", "nothing fun", "lost interest"],
            "Guilty Feelings": ["guilt", "guilty", "blame myself", "my fault"],
            "Punishment Feelings": ["deserve punishment", "should be punished"],
            "Self-Dislike": ["hate myself", "dislike myself", "don't like who I am"],
            "Self-Criticalness": ["self-critical", "too hard on myself", "judge myself"],
            "Suicidal Thoughts or Wishes": ["suicide", "suicidal", "end it all", "not worth living"],
            "Crying": ["crying", "cry", "tears", "weep"],
            "Agitation": ["agitated", "restless", "can't sit still"],
            "Loss of Interest": ["lost interest", "don't care", "nothing matters"],
            "Indecisiveness": ["indecisive", "can't decide", "uncertain"],
            "Worthlessness": ["worthless", "no value", "useless"],
            "Loss of Energy": ["no energy", "tired", "exhausted", "fatigue"],
            "Changes in Sleeping Pattern": ["sleep", "insomnia", "oversleeping", "can't sleep"],
            "Irritability": ["irritable", "irritated", "annoyed", "angry"],
            "Changes in Appetite": ["appetite", "eating", "not hungry", "overeating"],
            "Concentration Difficulty": ["concentrate", "focus", "difficulty thinking"],
            "Tiredness or Fatigue": ["tired", "fatigue", "exhausted", "worn out"],
            "Loss of Interest in Sex": ["sex", "intimacy", "no interest"]
        }
        return patterns
    
    def analyze_conversation(self, conversation_history: List[Dict]) -> DepressionPrediction:
        """
        Analyze conversation and predict depression.
        
        Args:
            conversation_history: List of messages with 'role' and 'message'
            
        Returns:
            DepressionPrediction object
        """
        # Extract all assistant messages (persona responses)
        persona_messages = [
            msg['message'] for msg in conversation_history 
            if msg['role'] == 'assistant'
        ]
        
        if not persona_messages:
            return DepressionPrediction(
                bdi_score=0,
                key_symptoms=[],
                confidence=0.0,
                turn_count=len(conversation_history)
            )
        
        # Combine all persona messages
        full_text = " ".join(persona_messages).lower()
        
        # Detect symptoms
        detected_symptoms = self._detect_symptoms(full_text, persona_messages)
        
        # Calculate BDI-II score
        bdi_score = self._calculate_bdi_score(detected_symptoms, full_text)
        
        # Get top symptoms
        key_symptoms = self._get_top_symptoms(detected_symptoms)
        
        # Calculate confidence
        confidence = self._calculate_confidence(detected_symptoms, bdi_score)
        
        return DepressionPrediction(
            bdi_score=bdi_score,
            key_symptoms=key_symptoms,
            confidence=confidence,
            turn_count=len(conversation_history)
        )
    
    def _detect_symptoms(self, full_text: str, messages: List[str]) -> Dict[str, float]:
        """Detect symptoms with confidence scores."""
        symptom_scores = {}
        
        for symptom in self.bdi_symptoms:
            patterns = self.symptom_patterns.get(symptom, [])
            score = 0.0
            
            # Keyword matching
            for pattern in patterns:
                if pattern in full_text:
                    score += 0.3
            
            # Semantic similarity (using sentence transformer)
            if patterns:
                try:
                    symptom_embeddings = self.semantic_model.encode(patterns)
                    message_embeddings = self.semantic_model.encode(messages)
                    
                    # Calculate max similarity
                    similarities = np.max(
                        np.dot(message_embeddings, symptom_embeddings.T),
                        axis=1
                    )
                    max_similarity = float(np.max(similarities))
                    score += max_similarity * 0.5
                except:
                    pass
            
            # Negative sentiment indicators
            negative_words = ["not", "no", "never", "can't", "don't", "won't"]
            for msg in messages:
                msg_lower = msg.lower()
                for pattern in patterns:
                    if pattern in msg_lower:
                        # Check for negation
                        for neg_word in negative_words:
                            if neg_word in msg_lower[max(0, msg_lower.find(pattern)-20):msg_lower.find(pattern)]:
                                score -= 0.2
                                break
            
            symptom_scores[symptom] = max(0.0, min(1.0, score))
        
        return symptom_scores
    
    def _calculate_bdi_score(self, symptom_scores: Dict[str, float], text: str) -> int:
        """
        Calculate BDI-II score (0-63).
        
        BDI-II scoring:
        - Each symptom can contribute 0-3 points
        - Total score ranges from 0-63
        """
        base_score = 0
        
        # Count depression indicators
        indicator_count = sum(1 for indicator in self.depression_indicators if indicator in text)
        base_score += min(indicator_count * 2, 20)
        
        # Add symptom scores (each symptom contributes proportionally)
        total_symptom_score = sum(symptom_scores.values())
        symptom_contribution = min(total_symptom_score * 2, 30)
        
        # Add severity indicators
        severity_keywords = ["very", "extremely", "completely", "totally", "always"]
        severity_count = sum(1 for keyword in severity_keywords if keyword in text)
        severity_contribution = min(severity_count * 1.5, 13)
        
        total_score = int(base_score + symptom_contribution + severity_contribution)
        return min(63, max(0, total_score))
    
    def _get_top_symptoms(self, symptom_scores: Dict[str, float]) -> List[str]:
        """Get top symptoms up to max_symptoms."""
        sorted_symptoms = sorted(
            symptom_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_symptoms = [
            symptom for symptom, score in sorted_symptoms
            if score >= self.symptom_threshold
        ][:self.max_symptoms]
        
        return top_symptoms
    
    def _calculate_confidence(self, symptom_scores: Dict[str, float], bdi_score: int) -> float:
        """Calculate confidence in prediction."""
        # Confidence based on symptom detection strength
        avg_symptom_score = np.mean(list(symptom_scores.values())) if symptom_scores else 0.0
        
        # Confidence based on BDI score (moderate scores are less certain)
        if bdi_score < 10:
            bdi_confidence = 0.7  # Low depression - moderate confidence
        elif bdi_score > 40:
            bdi_confidence = 0.9  # High depression - high confidence
        else:
            bdi_confidence = 0.6  # Moderate - lower confidence
        
        # Combine confidences
        confidence = (avg_symptom_score * 0.5 + bdi_confidence * 0.5)
        return float(np.clip(confidence, 0.0, 1.0))
    
    def should_stop_early(self, prediction: DepressionPrediction) -> bool:
        """
        Determine if we should stop early based on confidence.
        
        Args:
            prediction: Current depression prediction
            
        Returns:
            True if should stop early
        """
        min_turns = self.config['conversation'].get('min_turns', 3)
        
        if prediction.turn_count < min_turns:
            return False
        
        return prediction.confidence >= self.early_stop_threshold
