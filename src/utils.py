"""
Utility functions for eRisk 2026 Task 1.
"""

import os
from typing import Optional
from dotenv import load_dotenv


def load_config_from_env() -> dict:
    """Load configuration from environment variables."""
    load_dotenv()
    
    return {
        'HF_TOKEN': os.getenv('HF_TOKEN'),
        'BASE_MODEL': os.getenv('BASE_MODEL', 'meta-llama/Meta-Llama-3-8B-Instruct'),
        'HF_COLLECTION': os.getenv('HF_COLLECTION', 'irlab-udc/erisk2026'),
        'DEVICE': os.getenv('DEVICE', 'cuda'),
        'LOAD_IN_8BIT': os.getenv('LOAD_IN_8BIT', 'true').lower() == 'true',
    }


def get_conversation_strategies() -> list:
    """
    Get list of conversation strategies for automated runs.
    These strategies avoid directly asking about depression.
    """
    return [
        {
            "name": "general_wellbeing",
            "questions": [
                "How are you feeling today?",
                "What's been on your mind lately?",
                "How have things been going for you?",
                "What's your day been like?",
                "How are you doing overall?"
            ]
        },
        {
            "name": "sleep_and_energy",
            "questions": [
                "How have you been sleeping lately?",
                "How's your energy level been?",
                "Do you feel rested when you wake up?",
                "What's your sleep schedule been like?"
            ]
        },
        {
            "name": "activities_and_interests",
            "questions": [
                "What have you been up to recently?",
                "Any hobbies or activities you've been enjoying?",
                "What do you do to relax?",
                "Have you been doing anything fun lately?"
            ]
        },
        {
            "name": "social_and_relationships",
            "questions": [
                "How are things with friends and family?",
                "Have you been spending time with people?",
                "How do you feel about your relationships?",
                "Do you have people you can talk to?"
            ]
        },
        {
            "name": "thoughts_and_feelings",
            "questions": [
                "What kinds of thoughts have been going through your mind?",
                "How would you describe your mood lately?",
                "What's been weighing on you?",
                "How do you feel about yourself these days?"
            ]
        },
        {
            "name": "daily_life",
            "questions": [
                "How's your daily routine been?",
                "What's a typical day like for you?",
                "How do you spend your time?",
                "What's been challenging for you recently?"
            ]
        }
    ]


def format_bdi_severity(bdi_score: int) -> str:
    """
    Format BDI-II score into severity category.
    
    Args:
        bdi_score: BDI-II score (0-63)
        
    Returns:
        Severity category string
    """
    if bdi_score < 0 or bdi_score > 63:
        return "Out of range"
    if bdi_score < 14:
        return "Minimal"
    elif bdi_score < 20:
        return "Mild"
    elif bdi_score < 29:
        return "Moderate"
    else:
        return "Severe"


def validate_persona_id(persona_id: int) -> bool:
    """Validate persona ID is in valid range (1-20)."""
    return 1 <= persona_id <= 20


def validate_run_id(run_id: int) -> bool:
    """Validate run ID is in valid range (1-3)."""
    return 1 <= run_id <= 3
