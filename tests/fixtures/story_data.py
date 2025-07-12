"""
Test story data and fixtures for integration testing
"""

# Sample story conversations for testing
STORY_CONVERSATIONS = [
    {
        "user": "Let's start a fantasy story. Introduce a character named Elena, a skilled archer from the village of Millbrook.",
        "assistant": "Elena drew back her bowstring with practiced ease, her emerald eyes fixed on the distant target. The morning mist clung to the cobblestone streets of Millbrook, and she could hear the village awakening behind her. As the best archer in three counties, Elena had responsibilities that weighed heavily on her young shoulders. Today marked the beginning of what would become her greatest adventure."
    },
    {
        "user": "Have Elena meet a mysterious stranger who needs her help.",
        "assistant": "A cloaked figure emerged from the treeline beyond the practice range, moving with an urgent stride that immediately caught Elena's attention. The stranger's hood obscured their face, but Elena could see weathered hands that spoke of long travels and hard-won experience. 'Elena of Millbrook,' the stranger called out, their voice carrying a strange accent. 'I bring word from the Northern Kingdoms. The Shadow Wolves have returned, and only your arrows can pierce their cursed hide. Will you help us?'"
    },
    {
        "user": "Elena agrees but wants to know more about these Shadow Wolves.",
        "assistant": "Elena lowered her bow, studying the stranger carefully. 'I've heard tales of Shadow Wolves from my grandmother's stories,' she said, her voice steady despite the chill that ran down her spine. 'Creatures born of dark magic, immune to normal weapons. But those were just stories.' The stranger pushed back their hood, revealing an elderly woman with silver hair and eyes like storm clouds. 'I am Magistrate Thorne from the capital of Valorheim. The stories are real, child. These beasts have already destroyed two villages north of here. They feed on fear and grow stronger with each attack. Your enchanted arrows—the ones your grandfather crafted—are our only hope.'"
    },
    {
        "user": "Elena realizes she must retrieve her grandfather's arrows from the family vault.",
        "assistant": "Elena's breath caught in her throat. Her grandfather's arrows—she had almost forgotten about them, locked away in the old family vault beneath their cottage since his passing three years ago. 'The Silverwood arrows,' she whispered, remembering the stories he used to tell her about hunting creatures of darkness in his youth. 'But I've never used them. Grandfather said they were too dangerous, that they could corrupt the wielder if used without proper training.' Magistrate Thorne's expression grew grave. 'Then we must hope your bloodline and natural skill will protect you. Time is running short, Elena. The wolves are moving south, and they'll reach Millbrook within a fortnight.'"
    },
    {
        "user": "Elena descends into the family vault with Thorne.",
        "assistant": "The stone steps leading to the vault were slick with decades of moisture, and Elena's torch cast dancing shadows on the carved walls. Ancient runes glowed faintly as they passed, responding to her presence—evidence of the magical bloodline she'd never fully understood. Magistrate Thorne followed close behind, her breathing labored from the descent. The vault door stood before them, an imposing barrier of enchanted oak bound with silver. Elena placed her palm against the center lock, feeling the warm tingle of recognition magic. With a deep rumble, the door swung open, revealing a chamber filled with artifacts from her family's long history of protecting the realm from supernatural threats."
    }
]

# Character descriptions for consistency testing
CHARACTERS = {
    "Elena": {
        "name": "Elena",
        "occupation": "archer",
        "location": "Millbrook",
        "traits": ["skilled", "young", "responsible", "emerald eyes"],
        "equipment": ["bow", "arrows"],
        "family": ["grandfather (deceased)"],
        "abilities": ["archery", "magical bloodline"]
    },
    "Magistrate Thorne": {
        "name": "Magistrate Thorne", 
        "occupation": "magistrate",
        "location": "Valorheim",
        "traits": ["elderly", "silver hair", "storm cloud eyes", "weathered hands"],
        "background": ["from Northern Kingdoms capital"],
        "knowledge": ["Shadow Wolves", "magical creatures"]
    }
}

# Plot threads for continuity testing
PLOT_THREADS = {
    "shadow_wolves": {
        "name": "Shadow Wolf Threat",
        "status": "active",
        "description": "Magical creatures threatening villages",
        "elements": ["immune to normal weapons", "feed on fear", "destroyed two villages"],
        "urgency": "fortnight deadline"
    },
    "enchanted_arrows": {
        "name": "Grandfather's Silverwood Arrows",
        "status": "discovered",
        "description": "Family heirloom weapons effective against dark creatures",
        "elements": ["locked in vault", "dangerous to use", "require training"],
        "connection": "Elena's grandfather was a supernatural hunter"
    },
    "magical_bloodline": {
        "name": "Elena's Magical Heritage",
        "status": "emerging",
        "description": "Elena discovers her family's magical legacy",
        "elements": ["recognition magic", "vault responds to her", "protective bloodline"],
        "implications": "greater responsibilities and abilities"
    }
}

# Locations for world consistency
LOCATIONS = {
    "Millbrook": {
        "name": "Millbrook",
        "type": "village",
        "features": ["cobblestone streets", "practice range", "Elena's cottage"],
        "description": "Small village where Elena lives and practices archery"
    },
    "Valorheim": {
        "name": "Valorheim", 
        "type": "capital city",
        "location": "Northern Kingdoms",
        "description": "Capital city where Magistrate Thorne comes from"
    },
    "Family Vault": {
        "name": "Family Vault",
        "type": "underground chamber",
        "location": "beneath Elena's cottage",
        "features": ["stone steps", "carved walls", "ancient runes", "enchanted door"],
        "contents": ["grandfather's artifacts", "Silverwood arrows"]
    }
}

# Long conversation generator for performance testing
def generate_story_interactions(count: int) -> list:
    """Generate a list of story interactions for performance testing"""
    interactions = []
    
    # Start with base conversations
    for i, conv in enumerate(STORY_CONVERSATIONS):
        if i >= count:
            break
        interactions.append(conv)
    
    # Generate additional interactions if needed
    if count > len(STORY_CONVERSATIONS):
        remaining = count - len(STORY_CONVERSATIONS)
        
        action_templates = [
            "Elena practices with her {weapon} while thinking about {topic}.",
            "Magistrate Thorne shares more information about {threat} and {history}.",
            "Elena explores {location} and discovers {discovery}.",
            "A new challenge arises involving {character} and {obstacle}.",
            "Elena must decide between {choice1} and {choice2}.",
        ]
        
        for i in range(remaining):
            template_idx = i % len(action_templates)
            user_input = f"Continue the story. {action_templates[template_idx]}"
            
            # Simple response generation for testing
            assistant_response = f"Elena continued her journey, facing new challenges and growing stronger. The Shadow Wolves grew closer, but her determination never wavered. This is interaction {len(STORY_CONVERSATIONS) + i + 1} in the ongoing story."
            
            interactions.append({
                "user": user_input,
                "assistant": assistant_response
            })
    
    return interactions


# Memory retrieval test queries
MEMORY_QUERIES = [
    {
        "query": "Elena's appearance",
        "expected_elements": ["emerald eyes", "archer", "young"],
        "type": "character"
    },
    {
        "query": "Shadow Wolves characteristics", 
        "expected_elements": ["immune to normal weapons", "feed on fear", "dark magic"],
        "type": "creature"
    },
    {
        "query": "Millbrook village",
        "expected_elements": ["cobblestone streets", "practice range", "village"],
        "type": "location"
    },
    {
        "query": "grandfather's arrows",
        "expected_elements": ["Silverwood", "enchanted", "vault", "dangerous"],
        "type": "item"
    },
    {
        "query": "family magical bloodline",
        "expected_elements": ["recognition magic", "runes", "magical bloodline"],
        "type": "ability"
    }
]