"""
DOCX Table Generation and Formatting

This module handles the creation and formatting of tables in DOCX documents
with support for complex layouts, styling, and cell formatting.
"""

import logging
from typing import Optional, List, Tuple
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn

from ...models import ContentElement, TableData, TableRow, TableCell, TextStyle
from .styles import DocxStyleManager

logger = logging.getLogger(__name__)


class DocxTableHandler:
    """
    Handles table creation and formatting for DOCX documents.
    
    This class provides functionality to create tables with proper formatting,
    borders, cell styling, and support for merged cells.
    """
    
    def __init__(self):
        """Initialize the table handler."""
        self.style_manager = DocxStyleManager()
        
        # Default table settings
        self.default_border_size = Pt(0.5)
        self.default_cell_padding = Pt(4)
        self.default_border_color = "#000000"
        
        # Table alignment mappings
        self.alignment_map = {
            'left': WD_TABLE_ALIGNMENT.LEFT,
            'center': WD_TABLE_ALIGNMENT.CENTER,
            'right': WD_TABLE_ALIGNMENT.RIGHT
        }
        
        # Cell alignment mappings
        self.cell_alignment_map = {
            'left': WD_ALIGN_PARAGRAPH.LEFT,
            'center': WD_ALIGN_PARAGRAPH.CENTER,
            'right': WD_ALIGN_PARAGRAPH.RIGHT,
            'justify': WD_ALIGN_PARAGRAPH.JUSTIFY
        }
        
        # Vertical alignment mappings
        self.vertical_alignment_map = {
            'top': WD_ALIGN_VERTICAL.TOP,
            'middle': WD_ALIGN_VERTICAL.CENTER,
            'bottom': WD_ALIGN_VERTICAL.BOTTOM
        }
    
    async def add_table(self, document: Document, element: ContentElement) -> None:
        """
        Add a table element to the document.
        
        Args:
            document: DOCX document to add to
            element: Table element to add
        """
        try:
            if not element.table_data or not element.table_data.rows:
                logger.warning("Table element has no data or rows")
                return
            
            table_data = element.table_data
            
            # Calculate table dimensions
            max_cols = max(len(row.cells) for row in table_data.rows) if table_data.rows else 0
            if max_cols == 0:
                logger.warning("Table has no columns")
                return
            
            # Create table
            table = document.add_table(rows=len(table_data.rows), cols=max_cols)
            
            # Apply table-level styling
            await self._apply_table_style(table, table_data)
            
            # Populate table content
            await self._populate_table(table, table_data)
            
            # Apply table borders and formatting
            await self._apply_table_borders(table, table_data)
            
            # Add caption if provided
            if table_data.caption:
                await self._add_table_caption(document, table_data.caption)
            
            logger.debug(f"Added table: {len(table_data.rows)} rows x {max_cols} columns")
            
        except Exception as e:
            logger.error(f"Failed to add table: {str(e)}")
            # Fallback to simple text representation
            await self._add_table_fallback(document, element.table_data)
    
    async def _apply_table_style(self, table, table_data: TableData) -> None:
        """Apply table-level styling and properties."""
        try:
            # Set table alignment
            table.alignment = WD_TABLE_ALIGNMENT.CENTER  # Default center alignment
            
            # Apply column widths if specified
            if table_data.column_widths:
                await self._set_column_widths(table, table_data.column_widths)
            
            # Set table style properties
            table.allow_autofit = False
            
        except Exception as e:
            logger.warning(f"Failed to apply table style: {str(e)}")
    
    async def _populate_table(self, table, table_data: TableData) -> None:
        """Populate table with content and cell formatting."""
        try:
            for row_idx, table_row in enumerate(table_data.rows):
                docx_row = table.rows[row_idx]
                
                # Track merged cells to skip them
                col_offset = 0
                
                for cell_idx, table_cell in enumerate(table_row.cells):
                    actual_col_idx = cell_idx + col_offset
                    
                    if actual_col_idx >= len(docx_row.cells):
                        logger.warning(f"Cell index {actual_col_idx} exceeds table width")
                        break
                    
                    docx_cell = docx_row.cells[actual_col_idx]
                    
                    # Set cell content
                    docx_cell.text = table_cell.content
                    
                    # Apply cell formatting
                    await self._apply_cell_formatting(docx_cell, table_cell, table_row)
                    
                    # Handle cell merging
                    if table_cell.colspan > 1 or table_cell.rowspan > 1:
                        await self._merge_cells(table, row_idx, actual_col_idx, table_cell)
                        col_offset += table_cell.colspan - 1
                
        except Exception as e:
            logger.error(f"Failed to populate table: {str(e)}")
    
    async def _apply_cell_formatting(self, docx_cell, table_cell: TableCell, table_row: TableRow) -> None:
        """Apply formatting to individual table cell."""
        try:
            # Get cell paragraph (first paragraph in cell)
            cell_paragraph = docx_cell.paragraphs[0] if docx_cell.paragraphs else None
            if not cell_paragraph:
                return
            
            # Apply text alignment
            if table_cell.alignment in self.cell_alignment_map:
                cell_paragraph.alignment = self.cell_alignment_map[table_cell.alignment]
            
            # Apply vertical alignment
            if table_cell.vertical_alignment in self.vertical_alignment_map:
                docx_cell.vertical_alignment = self.vertical_alignment_map[table_cell.vertical_alignment]
            
            # Apply text styling
            text_style = table_cell.style or table_row.style
            if text_style and cell_paragraph.runs:
                self.style_manager.apply_text_style_to_run(cell_paragraph.runs[0], text_style)
            
            # Apply cell background color
            if table_cell.background_color or table_row.background_color:
                bg_color = table_cell.background_color or table_row.background_color
                await self._set_cell_background(docx_cell, bg_color)
            
            # Set cell padding
            await self._set_cell_padding(docx_cell)
            
            # Handle header styling
            if table_row.is_header:
                await self._apply_header_cell_style(cell_paragraph)
            
        except Exception as e:
            logger.warning(f"Failed to apply cell formatting: {str(e)}")
    
    async def _apply_header_cell_style(self, paragraph) -> None:
        """Apply special styling for header cells."""
        try:
            # Make header text bold
            for run in paragraph.runs:
                run.font.bold = True
                
            # Center align header text
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
        except Exception as e:
            logger.warning(f"Failed to apply header cell style: {str(e)}")
    
    async def _merge_cells(self, table, start_row: int, start_col: int, table_cell: TableCell) -> None:
        """Merge cells for colspan and rowspan."""
        try:
            if table_cell.colspan <= 1 and table_cell.rowspan <= 1:
                return  # No merging needed
            
            # Calculate end positions
            end_row = start_row + table_cell.rowspan - 1
            end_col = start_col + table_cell.colspan - 1
            
            # Ensure we don't exceed table bounds
            end_row = min(end_row, len(table.rows) - 1)
            end_col = min(end_col, len(table.rows[0].cells) - 1)
            
            # Get start and end cells
            start_cell = table.rows[start_row].cells[start_col]
            end_cell = table.rows[end_row].cells[end_col]
            
            # Merge cells
            start_cell.merge(end_cell)
            
            logger.debug(f"Merged cells from ({start_row},{start_col}) to ({end_row},{end_col})")
            
        except Exception as e:
            logger.warning(f"Failed to merge cells: {str(e)}")
    
    async def _apply_table_borders(self, table, table_data: TableData) -> None:
        """Apply borders to table."""
        try:
            # Set table borders
            border_size = self._convert_border_size(table_data.border_width)
            border_color = table_data.border_color or self.default_border_color
            
            # Apply borders to all cells
            for row in table.rows:
                for cell in row.cells:
                    await self._set_cell_borders(cell, table_data.border_style, border_size, border_color)
            
        except Exception as e:
            logger.warning(f"Failed to apply table borders: {str(e)}")
    
    async def _set_cell_borders(self, cell, border_style: str, border_size, border_color: str) -> None:
        """Set borders for individual cell."""
        try:
            # This requires XML manipulation in python-docx
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            
            # Create borders element
            tcBorders = OxmlElement('w:tcBorders')
            
            # Border sides
            sides = ['top', 'left', 'bottom', 'right']
            
            for side in sides:
                border = OxmlElement(f'w:{side}')
                border.set(qn('w:val'), 'single' if border_style == 'solid' else border_style)
                border.set(qn('w:sz'), str(int(border_size * 8)))  # Convert to eighths of a point
                border.set(qn('w:space'), '0')
                border.set(qn('w:color'), border_color.lstrip('#'))
                tcBorders.append(border)
            
            tcPr.append(tcBorders)
            
        except Exception as e:
            logger.warning(f"Failed to set cell borders: {str(e)}")
    
    async def _set_cell_background(self, cell, color: str) -> None:
        """Set cell background color."""
        try:
            # XML manipulation for background color
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            
            # Create shading element
            shading = OxmlElement('w:shd')
            shading.set(qn('w:val'), 'clear')
            shading.set(qn('w:color'), 'auto')
            shading.set(qn('w:fill'), color.lstrip('#'))
            
            tcPr.append(shading)
            
        except Exception as e:
            logger.warning(f"Failed to set cell background: {str(e)}")
    
    async def _set_cell_padding(self, cell) -> None:
        """Set cell padding."""
        try:
            # XML manipulation for cell margins
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            
            # Create margins element
            tcMar = OxmlElement('w:tcMar')
            
            # Set padding for all sides
            sides = ['top', 'left', 'bottom', 'right']
            padding_value = str(int(self.default_cell_padding.pt * 20))  # Convert to twentieths of a point
            
            for side in sides:
                margin = OxmlElement(f'w:{side}')
                margin.set(qn('w:w'), padding_value)
                margin.set(qn('w:type'), 'dxa')
                tcMar.append(margin)
            
            tcPr.append(tcMar)
            
        except Exception as e:
            logger.warning(f"Failed to set cell padding: {str(e)}")
    
    async def _set_column_widths(self, table, column_widths: List) -> None:
        """Set column widths for table."""
        try:
            for col_idx, width in enumerate(column_widths):
                if col_idx >= len(table.columns):
                    break
                    
                column = table.columns[col_idx]
                
                if isinstance(width, (int, float)):
                    # Assume width is in inches
                    column.width = Inches(width)
                elif isinstance(width, str):
                    if width.endswith('%'):
                        # Percentage width - approximate conversion
                        percentage = float(width.rstrip('%'))
                        column.width = Inches(6.5 * percentage / 100)  # Assume 6.5" page width
                    elif width.endswith('in'):
                        # Inches
                        column.width = Inches(float(width.rstrip('in')))
                    elif width.endswith('cm'):
                        # Centimeters
                        column.width = Inches(float(width.rstrip('cm')) / 2.54)
                    
        except Exception as e:
            logger.warning(f"Failed to set column widths: {str(e)}")
    
    def _convert_border_size(self, border_width: int) -> float:
        """Convert border width to points."""
        # Border width is typically in pixels, convert to points
        return max(0.5, border_width * 0.75)  # Approximate conversion
    
    async def _add_table_caption(self, document: Document, caption: str) -> None:
        """Add caption below table."""
        try:
            caption_paragraph = document.add_paragraph(f"Table: {caption}")
            caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Style caption
            for run in caption_paragraph.runs:
                run.font.italic = True
                run.font.size = Pt(9)
                
        except Exception as e:
            logger.warning(f"Failed to add table caption: {str(e)}")
    
    async def _add_table_fallback(self, document: Document, table_data: Optional[TableData]) -> None:
        """Add text-based table fallback when table creation fails."""
        try:
            if not table_data or not table_data.rows:
                document.add_paragraph("[Empty table]")
                return
            
            document.add_paragraph("Table (formatted as text):")
            
            for row in table_data.rows:
                row_text = " | ".join(cell.content for cell in row.cells)
                paragraph = document.add_paragraph(row_text)
                
                # Style header rows
                if row.is_header:
                    for run in paragraph.runs:
                        run.font.bold = True
                        
        except Exception as e:
            logger.error(f"Failed to add table fallback: {str(e)}")
            document.add_paragraph("[Table could not be displayed]")
    
    async def validate_table(self, table_data: TableData) -> Tuple[bool, Optional[str]]:
        """
        Validate table data structure.
        
        Args:
            table_data: Table data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not table_data.rows:
                return False, "Table has no rows"
            
            if not all(row.cells for row in table_data.rows):
                return False, "Table has rows with no cells"
            
            # Check for consistent column counts
            col_counts = [len(row.cells) for row in table_data.rows]
            if len(set(col_counts)) > 1:
                logger.warning(f"Table has inconsistent column counts: {col_counts}")
                # This is not necessarily an error, just a warning
            
            # Validate cell spans
            for row_idx, row in enumerate(table_data.rows):
                for cell_idx, cell in enumerate(row.cells):
                    if cell.colspan < 1 or cell.rowspan < 1:
                        return False, f"Invalid cell span at row {row_idx}, col {cell_idx}"
            
            return True, None
            
        except Exception as e:
            return False, f"Table validation error: {str(e)}"


# Test functionality
async def test_table_handler():
    """Test table handler functionality."""
    from docx import Document
    from ...models import ContentElement, ContentType, TableData, TableRow, TableCell
    
    # Create test document
    document = Document()
    handler = DocxTableHandler()
    
    try:
        # Create test table data
        table_data = TableData(
            rows=[
                TableRow(
                    cells=[
                        TableCell(content="Header 1"),
                        TableCell(content="Header 2"),
                        TableCell(content="Header 3")
                    ],
                    is_header=True
                ),
                TableRow(
                    cells=[
                        TableCell(content="Row 1, Col 1"),
                        TableCell(content="Row 1, Col 2"),
                        TableCell(content="Row 1, Col 3")
                    ]
                ),
                TableRow(
                    cells=[
                        TableCell(content="Row 2, Col 1"),
                        TableCell(content="Row 2, Col 2", colspan=2)  # Merged cell
                    ]
                )
            ],
            caption="Test Table",
            has_header=True
        )
        
        # Create table element
        table_element = ContentElement(
            type=ContentType.TABLE,
            table_data=table_data
        )
        
        # Test validation
        is_valid, error = await handler.validate_table(table_data)
        if not is_valid:
            print(f"Table validation failed: {error}")
            return False
        
        # Add table to document
        await handler.add_table(document, table_element)
        
        print("✅ Table handler test passed")
        return True
        
    except Exception as e:
        print(f"❌ Table handler test failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_table_handler())