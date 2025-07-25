"""
Style configuration system for document generation.

This module provides centralized style management, style inheritance,
and configuration utilities for consistent document formatting across
different output formats.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import copy
import logging

from ..models import (
    TextStyle,
    ContentType,
    StyleWeight,
    StyleEmphasis,
    TemplateStyle
)

logger = logging.getLogger(__name__)


class StyleScope(Enum):
    """Style application scope."""
    GLOBAL = "global"
    DOCUMENT = "document"
    SECTION = "section"
    ELEMENT = "element"


class StylePriority(Enum):
    """Style priority levels for conflict resolution."""
    SYSTEM = 0      # Built-in system styles (lowest priority)
    TEMPLATE = 1    # Template-defined styles
    THEME = 2       # Theme-based styles
    USER = 3        # User-defined styles
    INLINE = 4      # Inline styles (highest priority)


@dataclass
class StyleRule:
    """Style rule with conditions and properties."""
    name: str
    style: TextStyle
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: StylePriority = StylePriority.USER
    scope: StyleScope = StyleScope.ELEMENT
    applies_to: List[ContentType] = field(default_factory=list)
    description: Optional[str] = None
    
    def matches_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if style rule matches given context."""
        if not self.conditions:
            return True
        
        for condition_key, condition_value in self.conditions.items():
            context_value = context.get(condition_key)
            
            if isinstance(condition_value, dict):
                # Complex condition (operators)
                if not self._evaluate_complex_condition(context_value, condition_value):
                    return False
            else:
                # Simple equality check
                if context_value != condition_value:
                    return False
        
        return True
    
    def _evaluate_complex_condition(self, context_value: Any, condition: Dict[str, Any]) -> bool:
        """Evaluate complex condition with operators."""
        for operator, operand in condition.items():
            if operator == "equals":
                if context_value != operand:
                    return False
            elif operator == "not_equals":
                if context_value == operand:
                    return False
            elif operator == "in":
                if context_value not in operand:
                    return False
            elif operator == "not_in":
                if context_value in operand:
                    return False
            elif operator == "greater_than":
                if not (context_value and context_value > operand):
                    return False
            elif operator == "less_than":
                if not (context_value and context_value < operand):
                    return False
            elif operator == "contains":
                if not (context_value and operand in str(context_value)):
                    return False
            elif operator == "starts_with":
                if not (context_value and str(context_value).startswith(str(operand))):
                    return False
            elif operator == "ends_with":
                if not (context_value and str(context_value).endswith(str(operand))):
                    return False
        
        return True


@dataclass
class StyleSet:
    """Collection of related style rules."""
    name: str
    description: Optional[str] = None
    rules: List[StyleRule] = field(default_factory=list)
    parent_set: Optional[str] = None  # For inheritance
    version: str = "1.0"
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def add_rule(self, rule: StyleRule) -> None:
        """Add style rule to set."""
        # Remove existing rule with same name
        self.rules = [r for r in self.rules if r.name != rule.name]
        self.rules.append(rule)
    
    def get_rule(self, name: str) -> Optional[StyleRule]:
        """Get style rule by name."""
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None
    
    def get_rules_for_content_type(self, content_type: ContentType) -> List[StyleRule]:
        """Get all rules that apply to specific content type."""
        return [rule for rule in self.rules if content_type in rule.applies_to]
    
    def remove_rule(self, name: str) -> bool:
        """Remove style rule by name."""
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.name != name]
        return len(self.rules) < original_count


@dataclass
class StyleTheme:
    """Complete style theme with multiple style sets."""
    name: str
    description: Optional[str] = None
    style_sets: Dict[str, StyleSet] = field(default_factory=dict)
    color_palette: Dict[str, str] = field(default_factory=dict)
    font_stack: Dict[str, List[str]] = field(default_factory=dict)
    default_settings: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    
    def add_style_set(self, style_set: StyleSet) -> None:
        """Add style set to theme."""
        self.style_sets[style_set.name] = style_set
    
    def get_style_set(self, name: str) -> Optional[StyleSet]:
        """Get style set by name."""
        return self.style_sets.get(name)
    
    def get_all_rules(self) -> List[StyleRule]:
        """Get all style rules from all sets."""
        all_rules = []
        for style_set in self.style_sets.values():
            all_rules.extend(style_set.rules)
        return all_rules


class StyleResolver:
    """Resolves style conflicts and applies inheritance."""
    
    def __init__(self):
        self._cache: Dict[str, TextStyle] = {}
    
    def resolve_styles(
        self, 
        rules: List[StyleRule], 
        context: Dict[str, Any]
    ) -> Optional[TextStyle]:
        """
        Resolve multiple style rules into single style.
        
        Args:
            rules: List of style rules to resolve
            context: Context for condition evaluation
            
        Returns:
            Resolved TextStyle or None if no rules match
        """
        # Filter rules that match conditions
        matching_rules = [rule for rule in rules if rule.matches_conditions(context)]
        
        if not matching_rules:
            return None
        
        # Sort by priority (highest first)
        matching_rules.sort(key=lambda r: r.priority.value, reverse=True)
        
        # Create cache key
        cache_key = self._create_cache_key(matching_rules, context)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Merge styles starting from lowest priority
        resolved_style = TextStyle()
        
        for rule in reversed(matching_rules):  # Start with lowest priority
            resolved_style = self._merge_styles(resolved_style, rule.style)
        
        # Cache result
        self._cache[cache_key] = resolved_style
        
        return resolved_style
    
    def _merge_styles(self, base_style: TextStyle, override_style: TextStyle) -> TextStyle:
        """Merge two styles, with override taking precedence."""
        merged = copy.copy(base_style)
        
        # Override non-None values
        if override_style.font_family is not None:
            merged.font_family = override_style.font_family
        if override_style.font_size is not None:
            merged.font_size = override_style.font_size
        if override_style.weight != StyleWeight.NORMAL:
            merged.weight = override_style.weight
        if override_style.emphasis != StyleEmphasis.NONE:
            merged.emphasis = override_style.emphasis
        if override_style.color is not None:
            merged.color = override_style.color
        if override_style.background_color is not None:
            merged.background_color = override_style.background_color
        if override_style.line_height is not None:
            merged.line_height = override_style.line_height
        if override_style.letter_spacing is not None:
            merged.letter_spacing = override_style.letter_spacing
        
        return merged
    
    def _create_cache_key(self, rules: List[StyleRule], context: Dict[str, Any]) -> str:
        """Create cache key for style resolution."""
        rule_ids = [f"{rule.name}_{rule.priority.value}" for rule in rules]
        context_str = "_".join(f"{k}:{v}" for k, v in sorted(context.items()))
        return f"{'_'.join(rule_ids)}_{context_str}"
    
    def clear_cache(self) -> None:
        """Clear style resolution cache."""
        self._cache.clear()


class StyleManager:
    """Main style management system."""
    
    def __init__(self, config_directories: Optional[List[str]] = None):
        """
        Initialize style manager.
        
        Args:
            config_directories: Directories to search for style configurations
        """
        self.config_directories = config_directories or []
        self.themes: Dict[str, StyleTheme] = {}
        self.style_sets: Dict[str, StyleSet] = {}
        self.resolver = StyleResolver()
        self._active_theme: Optional[str] = None
        
        # Add default config directories
        current_dir = Path(__file__).parent.parent.parent.parent
        self.config_directories.extend([
            str(current_dir / "config" / "styles"),
            str(current_dir / "src" / "generation" / "config" / "builtin_styles")
        ])
        
        # Load built-in styles
        self._load_builtin_styles()
    
    async def initialize(self) -> None:
        """Initialize style manager by loading configurations."""
        await self._scan_config_directories()
        await self._setup_default_theme()
    
    def create_style_rule(
        self,
        name: str,
        style: TextStyle,
        applies_to: List[ContentType],
        conditions: Optional[Dict[str, Any]] = None,
        priority: StylePriority = StylePriority.USER,
        scope: StyleScope = StyleScope.ELEMENT
    ) -> StyleRule:
        """Create new style rule."""
        return StyleRule(
            name=name,
            style=style,
            applies_to=applies_to,
            conditions=conditions or {},
            priority=priority,
            scope=scope
        )
    
    def create_style_set(
        self,
        name: str,
        description: Optional[str] = None,
        parent_set: Optional[str] = None
    ) -> StyleSet:
        """Create new style set."""
        style_set = StyleSet(
            name=name,
            description=description,
            parent_set=parent_set
        )
        
        # Inherit from parent if specified
        if parent_set and parent_set in self.style_sets:
            parent = self.style_sets[parent_set]
            style_set.rules = copy.deepcopy(parent.rules)
        
        self.style_sets[name] = style_set
        return style_set
    
    def get_style_for_element(
        self,
        content_type: ContentType,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[TextStyle]:
        """
        Get resolved style for content element.
        
        Args:
            content_type: Type of content element
            context: Additional context for style resolution
            
        Returns:
            Resolved TextStyle or None
        """
        context = context or {}
        context['content_type'] = content_type
        
        # Collect applicable rules from active theme
        rules = []
        
        if self._active_theme and self._active_theme in self.themes:
            theme = self.themes[self._active_theme]
            all_rules = theme.get_all_rules()
            rules.extend([rule for rule in all_rules if content_type in rule.applies_to])
        
        # Add rules from individual style sets
        for style_set in self.style_sets.values():
            rules.extend(style_set.get_rules_for_content_type(content_type))
        
        return self.resolver.resolve_styles(rules, context)
    
    def apply_theme(self, theme_name: str) -> bool:
        """
        Apply style theme.
        
        Args:
            theme_name: Name of theme to apply
            
        Returns:
            True if theme was applied successfully
        """
        if theme_name not in self.themes:
            logger.warning(f"Theme '{theme_name}' not found")
            return False
        
        self._active_theme = theme_name
        self.resolver.clear_cache()  # Clear cache when theme changes
        
        logger.info(f"Applied theme '{theme_name}'")
        return True
    
    def get_active_theme(self) -> Optional[StyleTheme]:
        """Get currently active theme."""
        if self._active_theme:
            return self.themes.get(self._active_theme)
        return None
    
    def list_themes(self) -> List[Dict[str, Any]]:
        """List available themes."""
        return [
            {
                "name": name,
                "description": theme.description or "",
                "version": theme.version,
                "style_sets": list(theme.style_sets.keys()),
                "active": name == self._active_theme
            }
            for name, theme in self.themes.items()
        ]
    
    def export_style_set(self, style_set_name: str, file_path: str) -> bool:
        """Export style set to file."""
        if style_set_name not in self.style_sets:
            return False
        
        style_set = self.style_sets[style_set_name]
        
        try:
            # Convert to serializable format
            export_data = asdict(style_set)
            
            # Convert enums to strings
            for rule_data in export_data['rules']:
                if 'priority' in rule_data:
                    rule_data['priority'] = rule_data['priority'].name if hasattr(rule_data['priority'], 'name') else str(rule_data['priority'])
                if 'scope' in rule_data:
                    rule_data['scope'] = rule_data['scope'].name if hasattr(rule_data['scope'], 'name') else str(rule_data['scope'])
                if 'applies_to' in rule_data:
                    rule_data['applies_to'] = [ct.name if hasattr(ct, 'name') else str(ct) for ct in rule_data['applies_to']]
            
            # Write to file
            file_extension = Path(file_path).suffix.lower()
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_extension == '.yaml' or file_extension == '.yml':
                    yaml.dump(export_data, f, default_flow_style=False)
                else:
                    json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported style set '{style_set_name}' to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export style set '{style_set_name}': {e}")
            return False
    
    def import_style_set(self, file_path: str) -> Optional[str]:
        """Import style set from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_extension = Path(file_path).suffix.lower()
                if file_extension == '.yaml' or file_extension == '.yml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Convert back from serialized format
            style_set = self._deserialize_style_set(data)
            self.style_sets[style_set.name] = style_set
            
            logger.info(f"Imported style set '{style_set.name}' from {file_path}")
            return style_set.name
            
        except Exception as e:
            logger.error(f"Failed to import style set from {file_path}: {e}")
            return None
    
    def _load_builtin_styles(self) -> None:
        """Load built-in system styles."""
        # Create default style set
        default_set = StyleSet(
            name="default",
            description="Built-in default styles",
            version="1.0"
        )
        
        # Add common style rules
        default_rules = [
            StyleRule(
                name="heading_1",
                style=TextStyle(font_size=18, weight=StyleWeight.BOLD),
                applies_to=[ContentType.HEADING],
                conditions={"level": 1},
                priority=StylePriority.SYSTEM
            ),
            StyleRule(
                name="heading_2", 
                style=TextStyle(font_size=16, weight=StyleWeight.BOLD),
                applies_to=[ContentType.HEADING],
                conditions={"level": 2},
                priority=StylePriority.SYSTEM
            ),
            StyleRule(
                name="heading_3",
                style=TextStyle(font_size=14, weight=StyleWeight.BOLD),
                applies_to=[ContentType.HEADING],
                conditions={"level": 3},
                priority=StylePriority.SYSTEM
            ),
            StyleRule(
                name="paragraph",
                style=TextStyle(font_size=11, line_height=1.2),
                applies_to=[ContentType.PARAGRAPH, ContentType.TEXT],
                priority=StylePriority.SYSTEM
            ),
            StyleRule(
                name="code",
                style=TextStyle(font_family="Courier New", font_size=10),
                applies_to=[ContentType.CODE],
                priority=StylePriority.SYSTEM
            ),
            StyleRule(
                name="quote",
                style=TextStyle(font_size=11, emphasis=StyleEmphasis.ITALIC),
                applies_to=[ContentType.QUOTE],
                priority=StylePriority.SYSTEM
            )
        ]
        
        for rule in default_rules:
            default_set.add_rule(rule)
        
        self.style_sets["default"] = default_set
        
        # Create default theme
        default_theme = StyleTheme(
            name="default",
            description="Default system theme",
            color_palette={
                "primary": "#000000",
                "secondary": "#666666", 
                "accent": "#0066CC",
                "background": "#FFFFFF"
            },
            font_stack={
                "serif": ["Times New Roman", "serif"],
                "sans": ["Arial", "Helvetica", "sans-serif"],
                "mono": ["Courier New", "monospace"]
            }
        )
        
        default_theme.add_style_set(default_set)
        self.themes["default"] = default_theme
    
    async def _scan_config_directories(self) -> None:
        """Scan directories for style configuration files."""
        for directory in self.config_directories:
            if not Path(directory).exists():
                continue
            
            for file_path in Path(directory).rglob("*.yaml"):
                try:
                    style_set_name = self.import_style_set(str(file_path))
                    if style_set_name:
                        logger.info(f"Loaded style set '{style_set_name}' from {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load style config {file_path}: {e}")
            
            for file_path in Path(directory).rglob("*.json"):
                try:
                    style_set_name = self.import_style_set(str(file_path))
                    if style_set_name:
                        logger.info(f"Loaded style set '{style_set_name}' from {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load style config {file_path}: {e}")
    
    async def _setup_default_theme(self) -> None:
        """Setup default theme if none is active."""
        if not self._active_theme and "default" in self.themes:
            self.apply_theme("default")
    
    def _deserialize_style_set(self, data: Dict[str, Any]) -> StyleSet:
        """Deserialize style set from dictionary."""
        # Convert enum strings back to enums
        if 'rules' in data:
            for rule_data in data['rules']:
                if 'priority' in rule_data and isinstance(rule_data['priority'], str):
                    rule_data['priority'] = StylePriority[rule_data['priority']]
                if 'scope' in rule_data and isinstance(rule_data['scope'], str):
                    rule_data['scope'] = StyleScope[rule_data['scope']]
                if 'applies_to' in rule_data:
                    rule_data['applies_to'] = [ContentType[ct] if isinstance(ct, str) else ct 
                                               for ct in rule_data['applies_to']]
                
                # Convert style data
                if 'style' in rule_data:
                    style_data = rule_data['style']
                    if 'weight' in style_data and isinstance(style_data['weight'], str):
                        style_data['weight'] = StyleWeight[style_data['weight']]
                    if 'emphasis' in style_data and isinstance(style_data['emphasis'], str):
                        style_data['emphasis'] = StyleEmphasis[style_data['emphasis']]
                    
                    rule_data['style'] = TextStyle(**style_data)
                
                # Create style rule
                rule_data = {k: v for k, v in rule_data.items() if k != 'style'}
                rule_data['style'] = rule_data.get('style', TextStyle())
        
        return StyleSet(**data)


# Convenience functions for common operations

def create_heading_style(level: int, font_size: int, bold: bool = True) -> StyleRule:
    """Create heading style rule."""
    return StyleRule(
        name=f"heading_{level}",
        style=TextStyle(
            font_size=font_size,
            weight=StyleWeight.BOLD if bold else StyleWeight.NORMAL
        ),
        applies_to=[ContentType.HEADING],
        conditions={"level": level}
    )


def create_paragraph_style(font_size: int = 11, line_height: float = 1.2) -> StyleRule:
    """Create paragraph style rule."""
    return StyleRule(
        name="paragraph",
        style=TextStyle(font_size=font_size, line_height=line_height),
        applies_to=[ContentType.PARAGRAPH, ContentType.TEXT]
    )


def create_code_style(font_family: str = "Courier New", font_size: int = 10) -> StyleRule:
    """Create code style rule."""
    return StyleRule(
        name="code",
        style=TextStyle(font_family=font_family, font_size=font_size),
        applies_to=[ContentType.CODE]
    )


# Test function for validation
def test():
    """Test style configuration functionality."""
    
    # Create style manager
    manager = StyleManager()
    
    # Test creating style rule
    heading_rule = create_heading_style(1, 18, True)
    assert heading_rule.name == "heading_1"
    assert heading_rule.style.font_size == 18
    
    # Test style resolution
    style = manager.get_style_for_element(
        ContentType.HEADING,
        {"level": 1}
    )
    
    if not style:
        print("❌ Style resolution failed")
        return False
    
    # Test theme application
    success = manager.apply_theme("default")
    if not success:
        print("❌ Theme application failed")
        return False
    
    print("✅ Style configuration test passed")
    return True


if __name__ == "__main__":
    test()