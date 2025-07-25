"""
PDF Flowables Handler

This module handles complex PDF flowables including tables, custom elements,
and advanced formatting using ReportLab's flowables system.
"""

import logging
from typing import List, Optional, Any, Tuple, Dict
from reportlab.platypus import Table, TableStyle, Spacer, KeepTogether, Paragraph
from reportlab.platypus.flowables import Flowable, HRFlowable
from reportlab.lib import colors
from reportlab.lib.units import inch, cm, mm
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

from ...models import ContentElement, TableData, TableRow, TableCell, TextStyle

logger = logging.getLogger(__name__)


class PdfFlowableHandler:
    """
    Handles complex PDF flowables using ReportLab.
    
    This class manages the creation and styling of tables, custom flowables,
    and other complex document elements that require special handling.
    """
    
    def __init__(self):
        """Initialize the PDF flowables handler."""
        self.default_table_style = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]
    
    async def add_table(self, story: List, element: ContentElement) -> None:
        """
        Add a table element to the PDF story.
        
        Args:
            story: List of ReportLab flowables
            element: Table content element
        """
        try:
            if not element.table_data or not element.table_data.rows:
                logger.warning("Table element has no data")
                return
            
            table_data = element.table_data
            
            # Convert table data to format suitable for ReportLab
            rl_table_data = await self._convert_table_data(table_data)
            
            # Calculate column widths
            col_widths = await self._calculate_column_widths(table_data, rl_table_data)
            
            # Create ReportLab table
            rl_table = Table(rl_table_data, colWidths=col_widths, repeatRows=1 if table_data.has_header else 0)
            
            # Apply table styling
            table_style = await self._create_table_style(table_data)
            rl_table.setStyle(table_style)
            
            # Apply cell-specific formatting
            await self._apply_cell_formatting(rl_table, table_data)
            
            # Create table flowables
            flowables = []
            
            # Add spacing before table
            flowables.append(Spacer(1, 6))
            
            # Add table caption if provided
            if table_data.caption:
                caption_style = ParagraphStyle(
                    'TableCaption',
                    fontName='Helvetica-Bold',
                    fontSize=10,
                    alignment=TA_CENTER,
                    spaceAfter=6
                )
                caption = Paragraph(self._escape_text(table_data.caption), caption_style)
                flowables.append(caption)
            
            # Add the table
            flowables.append(rl_table)
            
            # Add spacing after table
            flowables.append(Spacer(1, 6))
            
            # Keep table elements together
            story.append(KeepTogether(flowables))
            
            logger.debug(f"Added table: {len(table_data.rows)} rows, {len(table_data.rows[0].cells) if table_data.rows else 0} columns")
            
        except Exception as e:
            logger.error(f"Failed to add table: {e}")
            # Fallback to simple text representation
            await self._add_table_fallback(story, element)
    
    async def _convert_table_data(self, table_data: TableData) -> List[List[Any]]:
        """
        Convert TableData to ReportLab table format.
        
        Args:
            table_data: Table data structure
            
        Returns:
            List of lists suitable for ReportLab Table
        """
        rl_data = []
        
        for row in table_data.rows:
            rl_row = []
            for cell in row.cells:
                # Handle cell spanning
                if cell.colspan > 1 or cell.rowspan > 1:
                    # ReportLab handles spanning differently, we'll need to track this
                    rl_row.append(self._escape_text(cell.content))
                else:
                    rl_row.append(self._escape_text(cell.content))
            rl_data.append(rl_row)
        
        return rl_data
    
    async def _calculate_column_widths(
        self, 
        table_data: TableData, 
        rl_data: List[List[Any]]
    ) -> Optional[List[float]]:
        """
        Calculate optimal column widths for the table.
        
        Args:
            table_data: Original table data
            rl_data: ReportLab table data
            
        Returns:
            List of column widths or None for auto-sizing
        """
        if table_data.column_widths:
            # Use specified column widths
            widths = []
            for width in table_data.column_widths:
                if isinstance(width, str) and width.endswith('%'):
                    # Convert percentage to points (assume 6 inch page width)
                    percentage = float(width[:-1]) / 100
                    widths.append(6 * inch * percentage)
                elif isinstance(width, (int, float)):
                    # Assume points or convert
                    widths.append(width if width > 50 else width * inch)
                else:
                    widths.append(None)  # Auto-size this column
            return widths
        
        # Auto-calculate based on content length
        if not rl_data:
            return None
        
        num_columns = len(rl_data[0])
        content_lengths = [0] * num_columns
        
        # Calculate average content length per column
        for row in rl_data:
            for i, cell in enumerate(row):
                if i < len(content_lengths):
                    content_lengths[i] = max(content_lengths[i], len(str(cell)))
        
        # Convert to relative widths
        total_length = sum(content_lengths)
        if total_length == 0:
            return None
        
        available_width = 6 * inch  # Assume 6 inches available width
        widths = []
        for length in content_lengths:
            relative_width = (length / total_length) * available_width
            widths.append(max(relative_width, 0.5 * inch))  # Minimum 0.5 inch per column
        
        return widths
    
    async def _create_table_style(self, table_data: TableData) -> TableStyle:
        """
        Create ReportLab TableStyle from table data.
        
        Args:
            table_data: Table data configuration
            
        Returns:
            ReportLab TableStyle
        """
        style_commands = []
        
        # Basic grid and alignment
        style_commands.extend([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ])
        
        # Apply padding
        padding = table_data.cell_padding or 5
        style_commands.extend([
            ('LEFTPADDING', (0, 0), (-1, -1), padding),
            ('RIGHTPADDING', (0, 0), (-1, -1), padding),
            ('TOPPADDING', (0, 0), (-1, -1), padding),
            ('BOTTOMPADDING', (0, 0), (-1, -1), padding),
        ])
        
        # Apply borders
        if table_data.border_width > 0:
            border_color = self._parse_color(table_data.border_color)
            style_commands.append(('GRID', (0, 0), (-1, -1), table_data.border_width, border_color))
        
        # Header row styling
        if table_data.has_header and table_data.rows:
            style_commands.extend([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
            ])
        
        # Alternating row colors for better readability
        if len(table_data.rows) > 1:
            start_row = 1 if table_data.has_header else 0
            for i in range(start_row, len(table_data.rows), 2):
                style_commands.append(('BACKGROUND', (0, i), (-1, i), colors.lightgrey))
        
        return TableStyle(style_commands)
    
    async def _apply_cell_formatting(self, rl_table: Table, table_data: TableData) -> None:
        """
        Apply cell-specific formatting to the ReportLab table.
        
        Args:
            rl_table: ReportLab Table object
            table_data: Table data with formatting information
        """
        try:
            additional_styles = []
            
            for row_idx, row in enumerate(table_data.rows):
                for col_idx, cell in enumerate(row.cells):
                    # Apply cell-specific styling
                    if cell.style:
                        if cell.style.color:
                            color = self._parse_color(cell.style.color)
                            additional_styles.append(('TEXTCOLOR', (col_idx, row_idx), (col_idx, row_idx), color))
                        
                        if cell.style.background_color:
                            bg_color = self._parse_color(cell.style.background_color)
                            additional_styles.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), bg_color))
                    
                    # Apply cell background color
                    if cell.background_color:
                        bg_color = self._parse_color(cell.background_color)
                        additional_styles.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), bg_color))
                    
                    # Apply cell alignment
                    if cell.alignment != 'left':
                        align_map = {'center': 'CENTER', 'right': 'RIGHT', 'left': 'LEFT'}
                        alignment = align_map.get(cell.alignment, 'LEFT')
                        additional_styles.append(('ALIGN', (col_idx, row_idx), (col_idx, row_idx), alignment))
                    
                    # Apply vertical alignment
                    if cell.vertical_alignment != 'top':
                        valign_map = {'middle': 'MIDDLE', 'bottom': 'BOTTOM', 'top': 'TOP'}
                        valignment = valign_map.get(cell.vertical_alignment, 'TOP')
                        additional_styles.append(('VALIGN', (col_idx, row_idx), (col_idx, row_idx), valignment))
                    
                    # Handle cell spanning
                    if cell.colspan > 1 or cell.rowspan > 1:
                        end_col = col_idx + cell.colspan - 1
                        end_row = row_idx + cell.rowspan - 1
                        additional_styles.append(('SPAN', (col_idx, row_idx), (end_col, end_row)))
            
            # Apply additional styles
            if additional_styles:
                current_style = rl_table.getStyle()._cmds
                current_style.extend(additional_styles)
                rl_table.setStyle(TableStyle(current_style))
                
        except Exception as e:
            logger.error(f"Failed to apply cell formatting: {e}")
    
    async def _add_table_fallback(self, story: List, element: ContentElement) -> None:
        """
        Add a simple text representation of the table as fallback.
        
        Args:
            story: List of ReportLab flowables
            element: Table content element
        """
        try:
            if not element.table_data or not element.table_data.rows:
                return
            
            styles = getSampleStyleSheet()
            
            # Add table caption if available
            if element.table_data.caption:
                story.append(Paragraph(f"<b>{element.table_data.caption}</b>", styles['Normal']))
                story.append(Spacer(1, 6))
            
            # Convert table to simple text format
            table_text = ""
            for row in element.table_data.rows:
                row_text = " | ".join([cell.content for cell in row.cells])
                table_text += f"{row_text}\n"
                if element.table_data.has_header and row == element.table_data.rows[0]:
                    # Add separator after header
                    separator = " | ".join(["-" * len(cell.content) for cell in row.cells])
                    table_text += f"{separator}\n"
            
            # Add as preformatted text
            table_style = ParagraphStyle(
                'TableFallback',
                parent=styles['Normal'],
                fontName='Courier',
                fontSize=9,
                leftIndent=10,
                spaceBefore=6,
                spaceAfter=6
            )
            
            story.append(Paragraph(self._escape_text(table_text), table_style))
            
        except Exception as e:
            logger.error(f"Failed to add table fallback: {e}")
    
    def create_custom_flowable(self, draw_function, width: float, height: float) -> Flowable:
        """
        Create a custom flowable with a drawing function.
        
        Args:
            draw_function: Function that takes (canvas, width, height) and draws content
            width: Flowable width in points
            height: Flowable height in points
            
        Returns:
            Custom Flowable object
        """
        class CustomFlowable(Flowable):
            def __init__(self, draw_func, w, h):
                Flowable.__init__(self)
                self.draw_func = draw_func
                self.width = w
                self.height = h
            
            def draw(self):
                self.draw_func(self.canv, self.width, self.height)
        
        return CustomFlowable(draw_function, width, height)
    
    def create_horizontal_rule(
        self, 
        width: str = "100%", 
        thickness: float = 1, 
        color: Any = colors.black,
        space_before: float = 6,
        space_after: float = 6
    ) -> HRFlowable:
        """
        Create a horizontal rule (divider line).
        
        Args:
            width: Width specification
            thickness: Line thickness in points
            color: Line color
            space_before: Space before the rule
            space_after: Space after the rule
            
        Returns:
            HRFlowable object
        """
        return HRFlowable(
            width=width,
            thickness=thickness,
            color=color,
            spaceBefore=space_before,
            spaceAfter=space_after
        )
    
    def create_text_box(
        self, 
        text: str, 
        width: float, 
        height: float,
        style: Optional[ParagraphStyle] = None,
        border: bool = True,
        background_color: Optional[Any] = None
    ) -> Flowable:
        """
        Create a text box flowable.
        
        Args:
            text: Text content
            width: Box width in points
            height: Box height in points
            style: Text style
            border: Whether to draw border
            background_color: Background color
            
        Returns:
            Custom text box flowable
        """
        def draw_text_box(canvas, w, h):
            # Draw background
            if background_color:
                canvas.setFillColor(background_color)
                canvas.rect(0, 0, w, h, fill=1, stroke=0)
            
            # Draw border
            if border:
                canvas.setStrokeColor(colors.black)
                canvas.setLineWidth(1)
                canvas.rect(0, 0, w, h, fill=0, stroke=1)
            
            # Draw text (simplified - would need proper text wrapping)
            canvas.setFillColor(colors.black)
            canvas.setFont('Helvetica', 10)
            text_lines = text.split('\n')
            y_offset = h - 15  # Start from top
            
            for line in text_lines[:int(h//12)]:  # Limit lines to fit
                canvas.drawString(5, y_offset, line[:int(w//6)])  # Rough character limit
                y_offset -= 12
        
        return self.create_custom_flowable(draw_text_box, width, height)
    
    def _escape_text(self, text: str) -> str:
        """Escape text for safe use in ReportLab elements."""
        if not text:
            return ""
        
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        return text
    
    def _parse_color(self, color_str: str) -> Any:
        """Parse color string to ReportLab color object."""
        try:
            if color_str.startswith('#'):
                # Hex color
                if len(color_str) == 7:  # #RRGGBB
                    r = int(color_str[1:3], 16) / 255.0
                    g = int(color_str[3:5], 16) / 255.0
                    b = int(color_str[5:7], 16) / 255.0
                    return colors.Color(r, g, b)
                elif len(color_str) == 4:  # #RGB
                    r = int(color_str[1], 16) / 15.0
                    g = int(color_str[2], 16) / 15.0
                    b = int(color_str[3], 16) / 15.0
                    return colors.Color(r, g, b)
            else:
                # Named color
                return getattr(colors, color_str.lower(), colors.black)
        
        except (ValueError, AttributeError):
            return colors.black
    
    def get_table_statistics(self, table_data: TableData) -> Dict[str, Any]:
        """
        Get statistics about a table for optimization.
        
        Args:
            table_data: Table data to analyze
            
        Returns:
            Dictionary with table statistics
        """
        if not table_data.rows:
            return {'rows': 0, 'columns': 0, 'cells': 0}
        
        row_count = len(table_data.rows)
        col_count = len(table_data.rows[0].cells) if table_data.rows else 0
        cell_count = sum(len(row.cells) for row in table_data.rows)
        
        # Calculate content statistics
        total_chars = sum(
            len(cell.content) 
            for row in table_data.rows 
            for cell in row.cells
        )
        
        avg_chars_per_cell = total_chars / cell_count if cell_count > 0 else 0
        
        # Check for spanning cells
        has_spanning = any(
            cell.colspan > 1 or cell.rowspan > 1
            for row in table_data.rows
            for cell in row.cells
        )
        
        return {
            'rows': row_count,
            'columns': col_count,
            'cells': cell_count,
            'total_characters': total_chars,
            'avg_chars_per_cell': avg_chars_per_cell,
            'has_header': table_data.has_header,
            'has_spanning_cells': has_spanning,
            'estimated_width': col_count * 1.5,  # Rough estimate in inches
            'complexity_score': self._calculate_table_complexity(table_data)
        }
    
    def _calculate_table_complexity(self, table_data: TableData) -> float:
        """Calculate complexity score for table rendering optimization."""
        score = 0.0
        
        # Base complexity
        score += len(table_data.rows) * 0.1
        score += len(table_data.rows[0].cells if table_data.rows else []) * 0.2
        
        # Spanning cells increase complexity
        for row in table_data.rows:
            for cell in row.cells:
                if cell.colspan > 1 or cell.rowspan > 1:
                    score += 0.5
        
        # Styling increases complexity
        if table_data.border_width > 1:
            score += 0.1
        
        return score