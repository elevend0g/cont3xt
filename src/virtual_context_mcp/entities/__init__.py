"""
Entity extraction and management for story content.

This module provides tools for extracting and managing story-specific entities
including characters, locations, plot elements, and themes from narrative text.
"""

from .story_entities import StoryEntity, StoryEntityExtractor

__all__ = ["StoryEntity", "StoryEntityExtractor"]