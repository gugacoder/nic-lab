"""
Template management system for document generation.

This module handles loading, caching, validation, and application of document
templates across different formats. It provides a centralized system for
managing corporate templates and custom formatting.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
from dataclasses import asdict
import asyncio
from datetime import datetime
import hashlib
import logging

from ..models import (
    DocumentTemplate, 
    TemplateStyle, 
    TemplateLayout,
    TextStyle,
    ContentType,
    DocumentContent,
    DocumentSection
)
from ..base import TemplateError

logger = logging.getLogger(__name__)


class TemplateCache:
    """In-memory cache for loaded templates."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, DocumentTemplate] = {}
        self._access_times: Dict[str, datetime] = {}
        self._file_hashes: Dict[str, str] = {}
    
    def get(self, template_id: str) -> Optional[DocumentTemplate]:
        """Get template from cache."""
        if template_id in self._cache:
            self._access_times[template_id] = datetime.now()
            return self._cache[template_id]
        return None
    
    def put(self, template: DocumentTemplate, file_path: Optional[str] = None) -> None:
        """Add template to cache."""
        if len(self._cache) >= self.max_size:
            self._evict_oldest()
        
        self._cache[template.id] = template
        self._access_times[template.id] = datetime.now()
        
        if file_path:
            self._file_hashes[template.id] = self._calculate_file_hash(file_path)
    
    def invalidate(self, template_id: str) -> None:
        """Remove template from cache."""
        self._cache.pop(template_id, None)
        self._access_times.pop(template_id, None)
        self._file_hashes.pop(template_id, None)
    
    def is_valid(self, template_id: str, file_path: str) -> bool:
        """Check if cached template is still valid."""
        if template_id not in self._cache:
            return False
        
        if template_id not in self._file_hashes:
            return True  # No file hash stored, assume valid
        
        current_hash = self._calculate_file_hash(file_path)
        return self._file_hashes[template_id] == current_hash
    
    def clear(self) -> None:
        """Clear all cached templates."""
        self._cache.clear()
        self._access_times.clear()
        self._file_hashes.clear()
    
    def _evict_oldest(self) -> None:
        """Remove the least recently used template."""
        if not self._access_times:
            return
        
        oldest_id = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self.invalidate(oldest_id)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of file for change detection."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""


class TemplateRegistry:
    """Registry of available templates."""
    
    def __init__(self):
        self._templates: Dict[str, str] = {}  # template_id -> file_path
        self._metadata: Dict[str, Dict[str, Any]] = {}  # template_id -> metadata
        self._format_templates: Dict[str, Set[str]] = {}  # format -> template_ids
    
    def register_template(
        self, 
        template_id: str, 
        file_path: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a template in the registry."""
        self._templates[template_id] = file_path
        self._metadata[template_id] = metadata or {}
        
        # Register by format if specified
        format_type = metadata.get('format_type', 'all') if metadata else 'all'
        if format_type not in self._format_templates:
            self._format_templates[format_type] = set()
        self._format_templates[format_type].add(template_id)
        
        logger.info(f"Registered template '{template_id}' from {file_path}")
    
    def unregister_template(self, template_id: str) -> None:
        """Unregister a template."""
        if template_id in self._templates:
            # Remove from format registries
            for format_set in self._format_templates.values():
                format_set.discard(template_id)
            
            del self._templates[template_id]
            self._metadata.pop(template_id, None)
            logger.info(f"Unregistered template '{template_id}'")
    
    def get_template_path(self, template_id: str) -> Optional[str]:
        """Get file path for template."""
        return self._templates.get(template_id)
    
    def get_template_metadata(self, template_id: str) -> Dict[str, Any]:
        """Get metadata for template."""
        return self._metadata.get(template_id, {})
    
    def list_templates(self, format_type: Optional[str] = None) -> List[str]:
        """List available template IDs."""
        if format_type:
            return list(self._format_templates.get(format_type, set()))
        return list(self._templates.keys())
    
    def list_templates_by_format(self) -> Dict[str, List[str]]:
        """List templates grouped by format."""
        return {fmt: list(ids) for fmt, ids in self._format_templates.items()}


class TemplateManager:
    """Main template management system."""
    
    def __init__(self, template_directories: Optional[List[str]] = None):
        """
        Initialize template manager.
        
        Args:
            template_directories: List of directories to search for templates
        """
        self.template_directories = template_directories or []
        self.cache = TemplateCache()
        self.registry = TemplateRegistry()
        self._default_template_id: Optional[str] = None
        
        # Add common template directories
        current_dir = Path(__file__).parent.parent.parent.parent
        self.template_directories.extend([
            str(current_dir / "templates"),
            str(current_dir / "src" / "generation" / "templates" / "builtin")
        ])
    
    async def initialize(self) -> None:
        """Initialize the template manager by scanning directories."""
        await self._scan_template_directories()
        await self._load_default_templates()
    
    async def load_template(self, template_id: str) -> DocumentTemplate:
        """
        Load template by ID.
        
        Args:
            template_id: Unique template identifier
            
        Returns:
            Loaded DocumentTemplate
            
        Raises:
            TemplateError: If template cannot be loaded
        """
        # Check cache first
        cached_template = self.cache.get(template_id)
        template_path = self.registry.get_template_path(template_id)
        
        if cached_template and template_path and self.cache.is_valid(template_id, template_path):
            return cached_template
        
        # Load from file
        if not template_path:
            raise TemplateError(f"Template '{template_id}' not found in registry")
        
        if not os.path.exists(template_path):
            raise TemplateError(f"Template file not found: {template_path}")
        
        try:
            template = await self._load_template_file(template_path)
            template.id = template_id  # Ensure ID matches registry
            
            # Cache the loaded template
            self.cache.put(template, template_path)
            
            logger.info(f"Loaded template '{template_id}' from {template_path}")
            return template
            
        except Exception as e:
            raise TemplateError(f"Failed to load template '{template_id}': {str(e)}")
    
    async def save_template(self, template: DocumentTemplate, file_path: Optional[str] = None) -> str:
        """
        Save template to file.
        
        Args:
            template: Template to save
            file_path: Optional custom file path
            
        Returns:
            Path where template was saved
        """
        if not file_path:
            # Generate default path
            template_dir = Path(self.template_directories[0]) if self.template_directories else Path("templates")
            template_dir.mkdir(parents=True, exist_ok=True)
            file_path = str(template_dir / f"{template.id}.yaml")
        
        try:
            template_data = asdict(template)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(template_data, f, default_flow_style=False, allow_unicode=True)
            
            # Register in registry
            self.registry.register_template(
                template.id, 
                file_path,
                {"name": template.name, "format_type": template.format_type}
            )
            
            # Update cache
            self.cache.put(template, file_path)
            
            logger.info(f"Saved template '{template.id}' to {file_path}")
            return file_path
            
        except Exception as e:
            raise TemplateError(f"Failed to save template '{template.id}': {str(e)}")
    
    async def create_template_from_document(
        self, 
        document: DocumentContent, 
        template_name: str,
        format_type: str = "all"
    ) -> DocumentTemplate:
        """
        Create template by extracting styles from document.
        
        Args:
            document: Document to extract styles from
            template_name: Name for new template
            format_type: Target format type
            
        Returns:
            Created DocumentTemplate
        """
        template = DocumentTemplate(
            name=template_name,
            format_type=format_type,
            description=f"Template extracted from document: {document.metadata.title or 'Untitled'}"
        )
        
        # Extract layout from document
        template.layout = TemplateLayout(
            page_size=document.page_size,
            page_orientation=document.page_orientation,
            margins=document.margins.copy()
        )
        
        # Extract styles from document elements
        styles = set()  # Use set to avoid duplicates
        
        for section in document.sections:
            if section.style:
                styles.add(self._create_template_style("section", section.style, [ContentType.HEADING]))
            
            for element in section.elements:
                if element.style:
                    style_name = f"{element.type.value}_style"
                    styles.add(self._create_template_style(style_name, element.style, [element.type]))
        
        template.styles = list(styles)
        
        # Set default metadata
        template.author = document.metadata.author
        template.created_date = datetime.now()
        
        return template
    
    async def apply_template(
        self, 
        document: DocumentContent, 
        template: DocumentTemplate
    ) -> DocumentContent:
        """
        Apply template to document content.
        
        Args:
            document: Document to modify
            template: Template to apply
            
        Returns:
            Modified document with template applied
        """
        # Apply layout settings
        document.page_size = template.layout.page_size
        document.page_orientation = template.layout.page_orientation
        document.margins = template.layout.margins.copy()
        
        # Apply global template metadata
        if template.company_name and not document.metadata.author:
            document.metadata.author = template.company_name
        
        # Apply styles to document elements
        style_map = {style.name: style for style in template.styles}
        
        for section in document.sections:
            await self._apply_styles_to_section(section, style_map)
        
        # Apply header/footer if specified
        if template.header_template:
            document.header_content = template.header_template
        if template.footer_template:
            document.footer_content = template.footer_template
        
        return document
    
    async def validate_template(self, template: DocumentTemplate) -> List[str]:
        """
        Validate template configuration.
        
        Args:
            template: Template to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not template.name:
            errors.append("Template name is required")
        
        if not template.format_type:
            errors.append("Template format_type is required")
        
        # Validate styles
        for i, style in enumerate(template.styles):
            if not style.name:
                errors.append(f"Style {i} missing name")
            if not style.applies_to:
                errors.append(f"Style '{style.name}' must specify content types it applies to")
        
        # Validate layout
        if template.layout.margins:
            for margin_name, value in template.layout.margins.items():
                if value < 0:
                    errors.append(f"Margin '{margin_name}' cannot be negative")
        
        # Check for template file if specified
        if template.template_file_path and not os.path.exists(template.template_file_path):
            errors.append(f"Template file not found: {template.template_file_path}")
        
        return errors
    
    def list_available_templates(self, format_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available templates with metadata.
        
        Args:
            format_type: Optional format filter
            
        Returns:
            List of template information dictionaries
        """
        template_ids = self.registry.list_templates(format_type)
        templates_info = []
        
        for template_id in template_ids:
            metadata = self.registry.get_template_metadata(template_id)
            templates_info.append({
                "id": template_id,
                "name": metadata.get("name", template_id),
                "format_type": metadata.get("format_type", "unknown"),
                "description": metadata.get("description", ""),
                "is_default": template_id == self._default_template_id
            })
        
        return sorted(templates_info, key=lambda x: x["name"])
    
    def set_default_template(self, template_id: str) -> None:
        """Set default template ID."""
        if template_id in self.registry.list_templates():
            self._default_template_id = template_id
            logger.info(f"Set default template to '{template_id}'")
        else:
            raise TemplateError(f"Template '{template_id}' not found")
    
    def get_default_template_id(self) -> Optional[str]:
        """Get default template ID."""
        return self._default_template_id
    
    async def _scan_template_directories(self) -> None:
        """Scan template directories for available templates."""
        for directory in self.template_directories:
            if not os.path.exists(directory):
                continue
            
            for file_path in Path(directory).rglob("*.yaml"):
                try:
                    template_id = file_path.stem
                    self.registry.register_template(template_id, str(file_path))
                except Exception as e:
                    logger.warning(f"Failed to register template {file_path}: {e}")
            
            for file_path in Path(directory).rglob("*.json"):
                try:
                    template_id = file_path.stem
                    self.registry.register_template(template_id, str(file_path))
                except Exception as e:
                    logger.warning(f"Failed to register template {file_path}: {e}")
    
    async def _load_template_file(self, file_path: str) -> DocumentTemplate:
        """Load template from file."""
        file_extension = Path(file_path).suffix.lower()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_extension == '.yaml' or file_extension == '.yml':
                data = yaml.safe_load(f)
            elif file_extension == '.json':
                data = json.load(f)
            else:
                raise TemplateError(f"Unsupported template file format: {file_extension}")
        
        return self._deserialize_template(data)
    
    def _deserialize_template(self, data: Dict[str, Any]) -> DocumentTemplate:
        """Deserialize template data from dictionary."""
        # Handle nested objects
        if 'layout' in data:
            data['layout'] = TemplateLayout(**data['layout'])
        
        if 'styles' in data:
            styles = []
            for style_data in data['styles']:
                if 'style' in style_data:
                    style_data['style'] = TextStyle(**style_data['style'])
                styles.append(TemplateStyle(**style_data))
            data['styles'] = styles
        
        # Handle datetime fields
        if 'created_date' in data and isinstance(data['created_date'], str):
            data['created_date'] = datetime.fromisoformat(data['created_date'])
        
        return DocumentTemplate(**data)
    
    def _create_template_style(
        self, 
        name: str, 
        text_style: TextStyle, 
        applies_to: List[ContentType]
    ) -> TemplateStyle:
        """Create template style from text style."""
        return TemplateStyle(
            name=name,
            style=text_style,
            applies_to=applies_to
        )
    
    async def _apply_styles_to_section(
        self, 
        section: DocumentSection, 
        style_map: Dict[str, TemplateStyle]
    ) -> None:
        """Apply template styles to section elements."""
        for element in section.elements:
            # Look for applicable styles
            for style_name, template_style in style_map.items():
                if element.type in template_style.applies_to:
                    if not element.style:
                        element.style = template_style.style
                    # Could implement style merging here if needed
        
        # Recursively apply to subsections
        for subsection in section.subsections:
            await self._apply_styles_to_section(subsection, style_map)
    
    async def _load_default_templates(self) -> None:
        """Load built-in default templates."""
        # This would typically load system default templates
        # For now, we'll create a simple default template
        default_template = DocumentTemplate(
            id="default",
            name="Default Template",
            format_type="all",
            description="Built-in default template"
        )
        
        # Set up default styles
        default_template.styles = [
            TemplateStyle(
                name="heading_1",
                style=TextStyle(font_size=18, weight="bold"),
                applies_to=[ContentType.HEADING]
            ),
            TemplateStyle(
                name="paragraph",
                style=TextStyle(font_size=11),
                applies_to=[ContentType.PARAGRAPH, ContentType.TEXT]
            )
        ]
        
        # Cache the default template
        self.cache.put(default_template)
        self.registry.register_template(
            "default", 
            "builtin:default",
            {"name": "Default Template", "format_type": "all"}
        )
        self.set_default_template("default")


# Test function for validation
async def test_loading():
    """Test template loading functionality."""
    manager = TemplateManager()
    await manager.initialize()
    
    # Test loading default template
    try:
        default_template = await manager.load_template("default")
        assert default_template.name == "Default Template"
        print("✅ Template loading test passed")
        return True
    except Exception as e:
        print(f"❌ Template loading test failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(test_loading())