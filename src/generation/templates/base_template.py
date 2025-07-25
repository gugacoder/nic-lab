"""
Base template class for document generation.

This module provides the base template class that all document templates
can inherit from, providing common functionality for template processing,
style application, and content transformation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import logging

from ..models import (
    DocumentContent, 
    DocumentTemplate,
    DocumentSection,
    ContentElement,
    ContentType,
    TextStyle,
    ImageData,
    TableData,
    ListData,
    DocumentMetadata
)
from ..base import TemplateError

logger = logging.getLogger(__name__)


@dataclass
class TemplateVariable:
    """Template variable definition."""
    name: str
    value: Any
    type: str = "string"  # string, number, boolean, date, list, dict
    required: bool = False
    default: Any = None
    description: Optional[str] = None


@dataclass
class TemplateContext:
    """Context for template processing."""
    variables: Dict[str, TemplateVariable] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_data: Dict[str, Any] = field(default_factory=dict)
    
    def set_variable(self, name: str, value: Any, var_type: str = "string", required: bool = False) -> None:
        """Set a template variable."""
        self.variables[name] = TemplateVariable(
            name=name,
            value=value,
            type=var_type,
            required=required
        )
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a template variable value."""
        var = self.variables.get(name)
        return var.value if var else default
    
    def has_variable(self, name: str) -> bool:
        """Check if variable exists."""
        return name in self.variables
    
    def validate_required_variables(self) -> List[str]:
        """Check that all required variables are set."""
        missing = []
        for var in self.variables.values():
            if var.required and var.value is None:
                missing.append(var.name)
        return missing


class BaseTemplate(ABC):
    """
    Base class for all document templates.
    
    This class provides common functionality for template processing,
    including variable substitution, conditional content, and style application.
    """
    
    def __init__(self, template_config: DocumentTemplate):
        """
        Initialize base template.
        
        Args:
            template_config: Template configuration
        """
        self.config = template_config
        self.context = TemplateContext()
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        self._setup_default_variables()
    
    @property
    def template_id(self) -> str:
        """Get template ID."""
        return self.config.id
    
    @property
    def template_name(self) -> str:
        """Get template name."""
        return self.config.name
    
    @property
    def supported_formats(self) -> List[str]:
        """Get supported output formats."""
        if self.config.format_type == "all":
            return ["docx", "pdf"]
        return [self.config.format_type]
    
    @abstractmethod
    async def process_document(
        self, 
        document: DocumentContent,
        context_vars: Optional[Dict[str, Any]] = None
    ) -> DocumentContent:
        """
        Process document with template.
        
        Args:
            document: Document to process
            context_vars: Additional context variables
            
        Returns:
            Processed document
        """
        pass
    
    @abstractmethod
    def get_template_variables(self) -> List[TemplateVariable]:
        """Get list of variables this template supports."""
        pass
    
    def set_context_variables(self, variables: Dict[str, Any]) -> None:
        """Set context variables for template processing."""
        for name, value in variables.items():
            self.context.set_variable(name, value)
    
    def validate_context(self) -> List[str]:
        """Validate template context."""
        errors = []
        
        # Check required variables
        missing_vars = self.context.validate_required_variables()
        if missing_vars:
            errors.extend([f"Missing required variable: {var}" for var in missing_vars])
        
        # Template-specific validation
        template_errors = self._validate_template_specific()
        errors.extend(template_errors)
        
        return errors
    
    async def apply_styles(self, document: DocumentContent) -> DocumentContent:
        """Apply template styles to document."""
        if not self.config.styles:
            return document
        
        # Create style lookup
        style_map = {}
        for template_style in self.config.styles:
            for content_type in template_style.applies_to:
                if content_type not in style_map:
                    style_map[content_type] = []
                style_map[content_type].append(template_style.style)
        
        # Apply styles to document sections
        for section in document.sections:
            await self._apply_styles_to_section(section, style_map)
        
        return document
    
    async def apply_layout(self, document: DocumentContent) -> DocumentContent:
        """Apply template layout to document."""
        layout = self.config.layout
        
        # Apply page settings
        document.page_size = layout.page_size
        document.page_orientation = layout.page_orientation
        document.margins = layout.margins.copy()
        
        # Apply column settings if needed
        if layout.column_count > 1:
            document.metadata.custom_properties["column_count"] = layout.column_count
            document.metadata.custom_properties["column_spacing"] = layout.column_spacing
        
        return document
    
    def process_template_string(self, template_string: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process template string with variable substitution.
        
        Supports variables like {variable_name} and {variable.property}.
        
        Args:
            template_string: String with template variables
            context: Additional context variables
            
        Returns:
            Processed string with variables substituted
        """
        if not template_string:
            return template_string
        
        # Merge context variables
        all_vars = {}
        for var in self.context.variables.values():
            all_vars[var.name] = var.value
        
        if context:
            all_vars.update(context)
        
        # Add built-in variables
        all_vars.update({
            "today": datetime.now().strftime("%Y-%m-%d"),
            "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "template_name": self.template_name,
            "template_id": self.template_id
        })
        
        # Process template variables
        return self._substitute_variables(template_string, all_vars)
    
    def evaluate_condition(self, condition: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Evaluate conditional expression.
        
        Supports basic conditions like:
        - {variable} == "value"
        - {variable} != "value"  
        - {variable} > 10
        - {variable} exists
        - {variable} not_exists
        
        Args:
            condition: Condition string to evaluate
            context: Additional context variables
            
        Returns:
            Boolean result of condition
        """
        if not condition:
            return True
        
        try:
            # Get all variables
            all_vars = {}
            for var in self.context.variables.values():
                all_vars[var.name] = var.value
            if context:
                all_vars.update(context)
            
            # Simple condition parsing
            condition = condition.strip()
            
            # Handle "exists" and "not_exists"
            if condition.endswith(" exists"):
                var_name = condition[:-7].strip()
                var_name = var_name.strip("{}")
                return var_name in all_vars and all_vars[var_name] is not None
            
            if condition.endswith(" not_exists"):
                var_name = condition[:-11].strip()
                var_name = var_name.strip("{}")
                return var_name not in all_vars or all_vars[var_name] is None
            
            # Handle comparison operators
            operators = ["==", "!=", ">=", "<=", ">", "<"]
            for op in operators:
                if op in condition:
                    left, right = condition.split(op, 1)
                    left = left.strip()
                    right = right.strip().strip('"\'')
                    
                    # Get left value
                    left_val = self._get_variable_value(left, all_vars)
                    
                    # Convert right value to appropriate type
                    right_val = self._convert_value(right, type(left_val) if left_val is not None else str)
                    
                    # Evaluate condition
                    if op == "==":
                        return left_val == right_val
                    elif op == "!=":
                        return left_val != right_val
                    elif op == ">":
                        return left_val > right_val if left_val is not None and right_val is not None else False
                    elif op == "<":
                        return left_val < right_val if left_val is not None and right_val is not None else False
                    elif op == ">=":
                        return left_val >= right_val if left_val is not None and right_val is not None else False
                    elif op == "<=":
                        return left_val <= right_val if left_val is not None and right_val is not None else False
            
            # If no operator found, treat as boolean variable
            var_name = condition.strip("{}")
            return bool(all_vars.get(var_name, False))
            
        except Exception as e:
            logger.warning(f"Failed to evaluate condition '{condition}': {e}")
            return False
    
    def get_header_content(self, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get processed header content."""
        if not self.config.header_template:
            return None
        return self.process_template_string(self.config.header_template, context)
    
    def get_footer_content(self, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Get processed footer content."""
        if not self.config.footer_template:
            return None
        return self.process_template_string(self.config.footer_template, context)
    
    def add_branding_elements(self, document: DocumentContent) -> DocumentContent:
        """Add branding elements to document."""
        if self.config.company_name and not document.metadata.author:
            document.metadata.author = self.config.company_name
        
        if self.config.logo_path:
            # Could add logo to document header or first section
            pass
        
        return document
    
    def _setup_default_variables(self) -> None:
        """Setup default template variables."""
        self.context.set_variable("company_name", self.config.company_name or "")
        self.context.set_variable("template_name", self.config.name)
        self.context.set_variable("template_version", self.config.version)
        self.context.set_variable("creation_date", datetime.now())
    
    def _validate_template_specific(self) -> List[str]:
        """Template-specific validation (override in subclasses)."""
        return []
    
    async def _apply_styles_to_section(
        self, 
        section: DocumentSection, 
        style_map: Dict[ContentType, List[TextStyle]]
    ) -> None:
        """Apply styles to section elements."""
        for element in section.elements:
            if element.type in style_map:
                styles = style_map[element.type]
                if styles and not element.style:
                    # Use first matching style
                    element.style = styles[0]
        
        # Apply to subsections
        for subsection in section.subsections:
            await self._apply_styles_to_section(subsection, style_map)
    
    def _substitute_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """Substitute variables in text."""
        def replace_var(match):
            var_name = match.group(1)
            
            # Handle nested properties (e.g., {user.name})
            if '.' in var_name:
                return self._get_nested_value(var_name, variables)
            
            return str(variables.get(var_name, match.group(0)))
        
        # Pattern for {variable_name} or {object.property}
        pattern = r'\{([^}]+)\}'
        return re.sub(pattern, replace_var, text)
    
    def _get_variable_value(self, var_expression: str, variables: Dict[str, Any]) -> Any:
        """Get variable value from expression."""
        var_name = var_expression.strip("{}")
        
        if '.' in var_name:
            return self._get_nested_value(var_name, variables)
        
        return variables.get(var_name)
    
    def _get_nested_value(self, var_path: str, variables: Dict[str, Any]) -> str:
        """Get nested variable value (e.g., user.name)."""
        parts = var_path.split('.')
        value = variables
        
        try:
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = getattr(value, part, None)
                
                if value is None:
                    return f"{{{var_path}}}"  # Return original if not found
            
            return str(value)
        except Exception:
            return f"{{{var_path}}}"
    
    def _convert_value(self, value_str: str, target_type: type) -> Any:
        """Convert string value to target type."""
        if target_type == int:
            try:
                return int(value_str)
            except ValueError:
                return 0
        elif target_type == float:
            try:
                return float(value_str)
            except ValueError:
                return 0.0
        elif target_type == bool:
            return value_str.lower() in ('true', '1', 'yes', 'on')
        else:
            return value_str


class StandardTemplate(BaseTemplate):
    """Standard template implementation with common functionality."""
    
    def __init__(self, template_config: DocumentTemplate):
        super().__init__(template_config)
        self._setup_standard_variables()
    
    async def process_document(
        self, 
        document: DocumentContent,
        context_vars: Optional[Dict[str, Any]] = None
    ) -> DocumentContent:
        """Process document with standard template."""
        if context_vars:
            self.set_context_variables(context_vars)
        
        # Apply template processing
        document = await self.apply_layout(document)
        document = await self.apply_styles(document)
        document = self.add_branding_elements(document)
        
        # Process template strings in content
        await self._process_content_strings(document)
        
        # Add header/footer
        document.header_content = self.get_header_content(context_vars)
        document.footer_content = self.get_footer_content(context_vars)
        
        return document
    
    def get_template_variables(self) -> List[TemplateVariable]:
        """Get standard template variables."""
        return [
            TemplateVariable("title", "", "string", True, description="Document title"),
            TemplateVariable("author", "", "string", False, description="Document author"),
            TemplateVariable("date", datetime.now(), "date", False, description="Document date"),
            TemplateVariable("company_name", "", "string", False, description="Company name"),
            TemplateVariable("department", "", "string", False, description="Department"),
            TemplateVariable("project", "", "string", False, description="Project name"),
        ]
    
    def _setup_standard_variables(self) -> None:
        """Setup standard template variables."""
        for var in self.get_template_variables():
            self.context.variables[var.name] = var
    
    async def _process_content_strings(self, document: DocumentContent) -> None:
        """Process template strings in document content."""
        context_dict = {var.name: var.value for var in self.context.variables.values()}
        
        # Process document metadata
        if document.metadata.title:
            document.metadata.title = self.process_template_string(document.metadata.title, context_dict)
        
        # Process sections
        for section in document.sections:
            await self._process_section_strings(section, context_dict)
    
    async def _process_section_strings(
        self, 
        section: DocumentSection, 
        context: Dict[str, Any]
    ) -> None:
        """Process template strings in section."""
        if section.title:
            section.title = self.process_template_string(section.title, context)
        
        for element in section.elements:
            if element.content:
                element.content = self.process_template_string(element.content, context)
            
            # Process image captions
            if element.image_data and element.image_data.caption:
                element.image_data.caption = self.process_template_string(
                    element.image_data.caption, context
                )
        
        # Process subsections
        for subsection in section.subsections:
            await self._process_section_strings(subsection, context)


# Test function for validation  
def test():
    """Test base template functionality."""
    from ..models import DocumentTemplate, TemplateLayout, TemplateStyle, TextStyle, ContentType
    
    # Create test template
    template_config = DocumentTemplate(
        id="test_template",
        name="Test Template",
        format_type="all",
        description="Test template for validation"
    )
    
    template_config.styles = [
        TemplateStyle(
            name="heading",
            style=TextStyle(font_size=16, weight="bold"),
            applies_to=[ContentType.HEADING]
        )
    ]
    
    # Create template instance
    template = StandardTemplate(template_config)
    
    # Test variable substitution
    template.context.set_variable("test_var", "Hello World")
    result = template.process_template_string("Test: {test_var}")
    
    if result != "Test: Hello World":
        print(f"❌ Template string processing failed: got '{result}'")
        return False
    
    # Test condition evaluation
    template.context.set_variable("count", 5)
    if not template.evaluate_condition("{count} > 3"):
        print("❌ Condition evaluation failed")
        return False
    
    print("✅ Base template test passed")
    return True


if __name__ == "__main__":
    test()