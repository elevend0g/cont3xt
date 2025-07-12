"""
Story-specific entity extraction and management.

This module provides functionality to extract and manage entities specific to 
creative writing and storytelling, including characters, locations, plot elements,
and themes from narrative text.
"""

import re
import spacy
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, Counter


# Dialogue attribution patterns
DIALOGUE_PATTERNS = [
    r'"([^"]*)"[,.]?\s*said\s+(\w+)',
    r'(\w+)\s+said[,.]?\s*"([^"]*)"',
    r'"([^"]*)"[,.]?\s*(\w+)\s+replied',
    r'"([^"]*)"[,.]?\s*(\w+)\s+whispered',
    r'"([^"]*)"[,.]?\s*(\w+)\s+shouted',
    r'"([^"]*)"[,.]?\s*(\w+)\s+asked',
    r'(\w+)\s+replied[,.]?\s*"([^"]*)"',
    r'(\w+)\s+whispered[,.]?\s*"([^"]*)"',
    r'(\w+)\s+shouted[,.]?\s*"([^"]*)"',
    r'(\w+)\s+asked[,.]?\s*"([^"]*)"',
]

# Location marker patterns
LOCATION_MARKERS = [
    r'\bin\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    r'\bat\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    r'\bthrough\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    r'\bto\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    r'\bfrom\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    r'\bnear\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
]

# Plot-relevant action indicators
PLOT_INDICATORS = [
    "decided", "realized", "discovered", "revealed", "confronted",
    "planned", "schemed", "betrayed", "rescued", "defeated",
    "fought", "escaped", "arrived", "departed", "met", "encountered",
    "learned", "understood", "remembered", "forgot", "died", "killed",
    "loved", "hated", "feared", "hoped", "dreamed", "awoke"
]

# Common character title patterns
CHARACTER_TITLES = [
    r'\b(?:Mr|Mrs|Ms|Dr|Professor|Captain|Lord|Lady|Sir|Dame|King|Queen|Prince|Princess)\.\s+([A-Z][a-z]+)',
    r'\b([A-Z][a-z]+)\s+(?:Jr|Sr|III|IV|V)\b',
]


@dataclass
class StoryEntity:
    """Represents a story entity with metadata."""
    name: str
    entity_type: str  # "character", "location", "plot_element", "theme"
    mentions: List[str] = field(default_factory=list)  # Different ways entity is referenced
    context: str = ""  # Surrounding context where found
    confidence: float = 1.0  # Confidence score for entity extraction
    attributes: Dict[str, str] = field(default_factory=dict)  # Additional attributes

    def add_mention(self, mention: str, context: str = "") -> None:
        """Add a new mention of this entity."""
        if mention not in self.mentions:
            self.mentions.append(mention)
        if context and not self.context:
            self.context = context

    def merge_with(self, other: 'StoryEntity') -> None:
        """Merge another entity into this one."""
        if other.entity_type == self.entity_type:
            for mention in other.mentions:
                if mention not in self.mentions:
                    self.mentions.append(mention)
            if other.context and not self.context:
                self.context = other.context
            self.attributes.update(other.attributes)


class StoryEntityExtractor:
    """Extracts story-specific entities from narrative text."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize spaCy model for NER."""
        self.nlp = spacy.load(model_name)
        self.dialogue_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in DIALOGUE_PATTERNS]
        self.location_patterns = [re.compile(pattern) for pattern in LOCATION_MARKERS]
        self.character_title_patterns = [re.compile(pattern) for pattern in CHARACTER_TITLES]
        
        # Cache for entity deduplication
        self._entity_cache: Dict[str, Dict[str, StoryEntity]] = {
            "characters": {},
            "locations": {},
            "plot_elements": {},
            "themes": {}
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[StoryEntity]]:
        """Extract all story entities from text."""
        entities = {
            "characters": self.extract_characters(text),
            "locations": self.extract_locations(text),
            "plot_elements": self.extract_plot_elements(text),
            "themes": self.extract_themes(text)
        }
        
        # Deduplicate entities
        for entity_type, entity_list in entities.items():
            entities[entity_type] = self._deduplicate_entities(entity_list, entity_type)
        
        return entities
    
    def extract_characters(self, text: str) -> List[StoryEntity]:
        """Extract character names and references."""
        characters = []
        doc = self.nlp(text)
        
        # Extract person entities from spaCy NER
        for ent in doc.ents:
            if ent.label_ == "PERSON" and len(ent.text.strip()) > 1:
                # Get surrounding context (50 characters before and after)
                start_idx = max(0, ent.start_char - 50)
                end_idx = min(len(text), ent.end_char + 50)
                context = text[start_idx:end_idx].strip()
                
                character = StoryEntity(
                    name=ent.text.strip(),
                    entity_type="character",
                    mentions=[ent.text.strip()],
                    context=context,
                    confidence=0.8  # High confidence for NER
                )
                characters.append(character)
        
        # Extract characters from dialogue attribution
        dialogue_speakers = self.extract_dialogue_speakers(text)
        for speaker, dialogues in dialogue_speakers.items():
            context = dialogues[0][:100] if dialogues else ""
            character = StoryEntity(
                name=speaker,
                entity_type="character",
                mentions=[speaker],
                context=context,
                confidence=0.9,  # Very high confidence for dialogue speakers
                attributes={"dialogue_count": str(len(dialogues))}
            )
            characters.append(character)
        
        # Extract characters with titles
        for pattern in self.character_title_patterns:
            for match in pattern.finditer(text):
                full_name = match.group(0)
                name = match.group(1)
                start_idx = max(0, match.start() - 50)
                end_idx = min(len(text), match.end() + 50)
                context = text[start_idx:end_idx].strip()
                
                character = StoryEntity(
                    name=name,
                    entity_type="character",
                    mentions=[full_name, name],
                    context=context,
                    confidence=0.85,
                    attributes={"has_title": "true"}
                )
                characters.append(character)
        
        return characters
    
    def extract_locations(self, text: str) -> List[StoryEntity]:
        """Extract location names."""
        locations = []
        doc = self.nlp(text)
        
        # Extract location entities from spaCy NER
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"] and len(ent.text.strip()) > 1:
                start_idx = max(0, ent.start_char - 50)
                end_idx = min(len(text), ent.end_char + 50)
                context = text[start_idx:end_idx].strip()
                
                location = StoryEntity(
                    name=ent.text.strip(),
                    entity_type="location",
                    mentions=[ent.text.strip()],
                    context=context,
                    confidence=0.7,
                    attributes={"ner_label": ent.label_}
                )
                locations.append(location)
        
        # Extract locations using pattern matching
        for pattern in self.location_patterns:
            for match in pattern.finditer(text):
                location_name = match.group(1)
                if len(location_name) > 2 and location_name.isalpha():
                    start_idx = max(0, match.start() - 50)
                    end_idx = min(len(text), match.end() + 50)
                    context = text[start_idx:end_idx].strip()
                    
                    location = StoryEntity(
                        name=location_name,
                        entity_type="location",
                        mentions=[location_name],
                        context=context,
                        confidence=0.6,
                        attributes={"extraction_method": "pattern"}
                    )
                    locations.append(location)
        
        return locations
    
    def extract_plot_elements(self, text: str) -> List[StoryEntity]:
        """Extract plot-relevant elements."""
        plot_elements = []
        doc = self.nlp(text)
        
        # Find sentences with plot indicators
        sentences = [sent.text.strip() for sent in doc.sents]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            found_indicators = [word for word in PLOT_INDICATORS if word in sentence_lower]
            
            if found_indicators:
                # Extract key verbs and subjects for plot elements
                sent_doc = self.nlp(sentence)
                for token in sent_doc:
                    if (token.pos_ == "VERB" and 
                        token.lemma_ in PLOT_INDICATORS and 
                        len(sentence) > 20):  # Filter very short sentences
                        
                        plot_element = StoryEntity(
                            name=f"plot_action_{token.lemma_}",
                            entity_type="plot_element",
                            mentions=[token.text],
                            context=sentence,
                            confidence=0.5,
                            attributes={
                                "action": token.lemma_,
                                "tense": token.tag_,
                                "sentence": sentence
                            }
                        )
                        plot_elements.append(plot_element)
        
        return plot_elements
    
    def extract_themes(self, text: str) -> List[StoryEntity]:
        """Extract thematic elements from text."""
        themes = []
        doc = self.nlp(text)
        
        # Simple thematic keyword extraction
        theme_keywords = {
            "love": ["love", "heart", "romance", "affection", "passion"],
            "death": ["death", "died", "kill", "murder", "grave", "funeral"],
            "power": ["power", "control", "authority", "command", "rule"],
            "betrayal": ["betray", "deceive", "lie", "cheat", "backstab"],
            "friendship": ["friend", "companion", "ally", "buddy", "pal"],
            "fear": ["fear", "terror", "afraid", "scared", "frightened"],
            "hope": ["hope", "dream", "wish", "aspire", "optimism"],
            "family": ["family", "mother", "father", "brother", "sister", "parent"]
        }
        
        text_lower = text.lower()
        for theme_name, keywords in theme_keywords.items():
            keyword_count = sum(text_lower.count(keyword) for keyword in keywords)
            if keyword_count >= 2:  # Theme must appear at least twice
                # Find a representative sentence
                sentences = [sent.text for sent in doc.sents]
                representative_sentence = ""
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in keywords):
                        representative_sentence = sentence
                        break
                
                theme = StoryEntity(
                    name=theme_name,
                    entity_type="theme",
                    mentions=keywords,
                    context=representative_sentence,
                    confidence=min(0.8, keyword_count / 10),  # Cap at 0.8
                    attributes={"keyword_count": str(keyword_count)}
                )
                themes.append(theme)
        
        return themes
    
    def extract_dialogue_speakers(self, text: str) -> Dict[str, List[str]]:
        """Map dialogue to speakers."""
        speakers = defaultdict(list)
        
        for pattern in self.dialogue_patterns:
            for match in pattern.finditer(text):
                groups = match.groups()
                if len(groups) == 2:
                    # Determine which group is the speaker and which is the dialogue
                    speaker_idx = 1 if groups[1].isalpha() and groups[1][0].isupper() else 0
                    dialogue_idx = 1 - speaker_idx
                    
                    speaker = groups[speaker_idx].strip()
                    dialogue = groups[dialogue_idx].strip()
                    
                    if speaker and len(speaker) > 1 and speaker.isalpha():
                        speakers[speaker].append(dialogue)
        
        return dict(speakers)
    
    def _deduplicate_entities(self, entities: List[StoryEntity], entity_type: str) -> List[StoryEntity]:
        """Deduplicate entities by normalizing names and merging similar ones."""
        if not entities:
            return entities
        
        # Group entities by normalized names
        normalized_groups = defaultdict(list)
        
        for entity in entities:
            normalized_name = self._normalize_name(entity.name)
            normalized_groups[normalized_name].append(entity)
        
        # Merge entities in each group
        deduplicated = []
        for group in normalized_groups.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Merge entities with highest confidence as base
                base_entity = max(group, key=lambda e: e.confidence)
                for other_entity in group:
                    if other_entity != base_entity:
                        base_entity.merge_with(other_entity)
                deduplicated.append(base_entity)
        
        return deduplicated
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity names for deduplication."""
        # Remove common prefixes/suffixes and normalize case
        normalized = name.strip().lower()
        
        # Remove titles
        for title in ["mr.", "mrs.", "ms.", "dr.", "professor", "captain", "lord", "lady", "sir", "dame"]:
            if normalized.startswith(title + " "):
                normalized = normalized[len(title):].strip()
        
        # Remove suffixes
        for suffix in [" jr.", " sr.", " iii", " iv", " v"]:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
        
        # Handle common nickname patterns
        nickname_map = {
            "johnny": "john",
            "jimmy": "james",
            "bobby": "robert",
            "billy": "william",
            "tommy": "thomas",
            "danny": "daniel",
            "mike": "michael",
            "dave": "david",
            "joe": "joseph",
            "bob": "robert",
            "bill": "william",
            "tom": "thomas",
            "dan": "daniel",
            "jim": "james",
            "jack": "john"
        }
        
        if normalized in nickname_map:
            normalized = nickname_map[normalized]
        
        return normalized