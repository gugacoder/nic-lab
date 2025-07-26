"""
Page Display Component

Handles page-based document rendering with support for multi-page documents,
page breaks, page navigation, and different display modes.
"""

import streamlit as st
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class PageDisplayMode(Enum):
    """Page display modes"""
    SINGLE_PAGE = "single"
    CONTINUOUS = "continuous"
    FACING_PAGES = "facing"
    THUMBNAIL_GRID = "thumbnails"


class PageSize(Enum):
    """Standard page sizes"""
    A4 = {"width": 794, "height": 1123, "name": "A4"}
    LETTER = {"width": 816, "height": 1056, "name": "Letter"}
    LEGAL = {"width": 816, "height": 1344, "name": "Legal"}
    A3 = {"width": 1123, "height": 1587, "name": "A3"}
    TABLOID = {"width": 1056, "height": 1632, "name": "Tabloid"}


@dataclass
class PageContent:
    """Content for a single page"""
    page_number: int
    content: str
    content_type: str = "html"  # "html", "markdown", "text"
    page_size: PageSize = PageSize.A4
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PageDisplayConfig:
    """Configuration for page display"""
    display_mode: PageDisplayMode = PageDisplayMode.CONTINUOUS
    page_size: PageSize = PageSize.A4
    show_page_numbers: bool = True
    show_page_breaks: bool = True
    show_page_shadows: bool = True
    page_margin: int = 96  # pixels (1 inch at 96 DPI)
    inter_page_spacing: int = 32  # pixels between pages
    thumbnail_columns: int = 4
    facing_page_gap: int = 20  # gap between facing pages
    enable_page_navigation: bool = True


class PageDisplay:
    """
    Page display component for multi-page document rendering.
    
    Provides various display modes including single page, continuous scroll,
    facing pages, and thumbnail grid views.
    """
    
    @staticmethod
    def render_pages(
        pages: List[PageContent],
        config: Optional[PageDisplayConfig] = None,
        current_page: int = 1,
        zoom_level: int = 100,
        container_key: str = "page_display"
    ) -> Dict[str, Any]:
        """
        Render multi-page document display.
        
        Args:
            pages: List of page content objects
            config: Page display configuration
            current_page: Currently selected page number
            zoom_level: Zoom level percentage
            container_key: Unique key for the container
            
        Returns:
            Dictionary with page interactions and state
        """
        if config is None:
            config = PageDisplayConfig()
        
        if not pages:
            st.warning("📄 No pages to display")
            return {}
        
        interactions = {}
        
        # Validate current page
        current_page = max(1, min(len(pages), current_page))
        
        # Render based on display mode
        if config.display_mode == PageDisplayMode.SINGLE_PAGE:
            interactions = PageDisplay._render_single_page(
                pages, config, current_page, zoom_level, container_key
            )
        elif config.display_mode == PageDisplayMode.CONTINUOUS:
            interactions = PageDisplay._render_continuous_pages(
                pages, config, zoom_level, container_key
            )
        elif config.display_mode == PageDisplayMode.FACING_PAGES:
            interactions = PageDisplay._render_facing_pages(
                pages, config, current_page, zoom_level, container_key
            )
        elif config.display_mode == PageDisplayMode.THUMBNAIL_GRID:
            interactions = PageDisplay._render_thumbnail_grid(
                pages, config, current_page, zoom_level, container_key
            )
        
        return interactions
    
    @staticmethod
    def _render_single_page(
        pages: List[PageContent],
        config: PageDisplayConfig,
        current_page: int,
        zoom_level: int,
        container_key: str
    ) -> Dict[str, Any]:
        """Render single page display mode"""
        interactions = {}
        
        # Get current page content
        page = pages[current_page - 1]
        
        # Page navigation
        if config.enable_page_navigation:
            nav_interactions = PageDisplay._render_page_navigation(
                current_page, len(pages), container_key
            )
            interactions.update(nav_interactions)
        
        # Render the page
        page_interactions = PageDisplay._render_single_page_content(
            page, config, zoom_level, f"{container_key}_page_{current_page}"
        )
        interactions.update(page_interactions)
        
        return interactions
    
    @staticmethod
    def _render_continuous_pages(
        pages: List[PageContent],
        config: PageDisplayConfig,
        zoom_level: int,
        container_key: str
    ) -> Dict[str, Any]:
        """Render continuous scroll display mode"""
        interactions = {}
        
        # Container for all pages
        st.markdown('<div class="continuous-pages-container">', unsafe_allow_html=True)
        
        for i, page in enumerate(pages):
            # Render each page
            page_interactions = PageDisplay._render_single_page_content(
                page, config, zoom_level, f"{container_key}_page_{i+1}"
            )
            
            # Add page break visualization between pages (except last)
            if i < len(pages) - 1 and config.show_page_breaks:
                PageDisplay._render_page_break(i + 1)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return interactions
    
    @staticmethod
    def _render_facing_pages(
        pages: List[PageContent],
        config: PageDisplayConfig,
        current_page: int,
        zoom_level: int,
        container_key: str
    ) -> Dict[str, Any]:
        """Render facing pages display mode"""
        interactions = {}
        
        # Calculate which pages to show
        if current_page % 2 == 1:  # Odd page number
            left_page_num = current_page
            right_page_num = current_page + 1 if current_page < len(pages) else None
        else:  # Even page number
            left_page_num = current_page - 1
            right_page_num = current_page
        
        # Page navigation
        if config.enable_page_navigation:
            nav_interactions = PageDisplay._render_page_navigation(
                current_page, len(pages), container_key
            )
            interactions.update(nav_interactions)
        
        # Render facing pages
        col1, col_gap, col2 = st.columns([1, 0.1, 1])
        
        with col1:
            if left_page_num <= len(pages):
                left_page = pages[left_page_num - 1]
                PageDisplay._render_single_page_content(
                    left_page, config, zoom_level, f"{container_key}_left_{left_page_num}"
                )
            else:
                st.markdown('<div class="empty-page"></div>', unsafe_allow_html=True)
        
        with col2:
            if right_page_num and right_page_num <= len(pages):
                right_page = pages[right_page_num - 1]
                PageDisplay._render_single_page_content(
                    right_page, config, zoom_level, f"{container_key}_right_{right_page_num}"
                )
            else:
                st.markdown('<div class="empty-page"></div>', unsafe_allow_html=True)
        
        return interactions
    
    @staticmethod
    def _render_thumbnail_grid(
        pages: List[PageContent],
        config: PageDisplayConfig,
        current_page: int,
        zoom_level: int,
        container_key: str
    ) -> Dict[str, Any]:
        """Render thumbnail grid display mode"""
        interactions = {}
        
        # Calculate grid layout
        cols_per_row = config.thumbnail_columns
        num_rows = math.ceil(len(pages) / cols_per_row)
        
        st.markdown('<div class="thumbnail-grid">', unsafe_allow_html=True)
        
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            
            for col_idx in range(cols_per_row):
                page_idx = row * cols_per_row + col_idx
                
                if page_idx < len(pages):
                    with cols[col_idx]:
                        page = pages[page_idx]
                        page_num = page_idx + 1
                        
                        # Thumbnail container
                        is_current = page_num == current_page
                        thumbnail_class = "thumbnail-page current" if is_current else "thumbnail-page"
                        
                        if st.button(
                            f"📄 Page {page_num}",
                            key=f"{container_key}_thumb_{page_num}",
                            help=f"Go to page {page_num}"
                        ):
                            interactions["page_selected"] = page_num
                        
                        # Mini preview (scaled down)
                        thumbnail_zoom = 25  # Fixed small zoom for thumbnails
                        PageDisplay._render_single_page_content(
                            page, config, thumbnail_zoom, 
                            f"{container_key}_thumb_content_{page_num}",
                            is_thumbnail=True
                        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return interactions
    
    @staticmethod
    def _render_single_page_content(
        page: PageContent,
        config: PageDisplayConfig,
        zoom_level: int,
        container_key: str,
        is_thumbnail: bool = False
    ) -> Dict[str, Any]:
        """Render content for a single page"""
        interactions = {}
        
        # Determine page size and classes
        page_size_info = page.page_size.value
        page_width = page_size_info["width"]
        page_height = page_size_info["height"]
        
        # CSS classes
        page_classes = ["document-page"]
        if page.page_size == PageSize.A4:
            page_classes.append("a4-size")
        elif page.page_size == PageSize.LETTER:
            page_classes.append("letter-size")
        elif page.page_size == PageSize.LEGAL:
            page_classes.append("legal-size")
        
        if is_thumbnail:
            page_classes.append("thumbnail")
        
        if config.show_page_shadows:
            page_classes.append("with-shadow")
        
        # Container with zoom
        zoom_style = f"transform: scale({zoom_level / 100}); transform-origin: top center;"
        
        st.markdown(
            f'<div class="{" ".join(page_classes)}" style="{zoom_style}">',
            unsafe_allow_html=True
        )
        
        # Page number display
        if config.show_page_numbers and not is_thumbnail:
            st.markdown(
                f'<div class="page-number">Page {page.page_number}</div>',
                unsafe_allow_html=True
            )
        
        # Page content
        st.markdown('<div class="document-content">', unsafe_allow_html=True)
        
        if page.content_type.lower() == "html":
            st.markdown(page.content, unsafe_allow_html=True)
        elif page.content_type.lower() == "markdown":
            st.markdown(page.content)
        else:
            st.text(page.content)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        return interactions
    
    @staticmethod
    def _render_page_navigation(
        current_page: int,
        total_pages: int,
        container_key: str
    ) -> Dict[str, Any]:
        """Render page navigation controls"""
        interactions = {}
        
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])
        
        # First page
        with nav_col1:
            if st.button("⏮️", key=f"{container_key}_first", disabled=current_page <= 1):
                interactions["page_changed"] = 1
        
        # Previous page
        with nav_col2:
            if st.button("⏪", key=f"{container_key}_prev", disabled=current_page <= 1):
                interactions["page_changed"] = current_page - 1
        
        # Page input
        with nav_col3:
            page_input_col1, page_input_col2 = st.columns([1, 1])
            
            with page_input_col1:
                new_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=current_page,
                    key=f"{container_key}_page_input",
                    label_visibility="collapsed"
                )
                if new_page != current_page:
                    interactions["page_changed"] = new_page
            
            with page_input_col2:
                st.markdown(f"of {total_pages}")
        
        # Next page
        with nav_col4:
            if st.button("⏩", key=f"{container_key}_next", disabled=current_page >= total_pages):
                interactions["page_changed"] = current_page + 1
        
        # Last page
        with nav_col5:
            if st.button("⏭️", key=f"{container_key}_last", disabled=current_page >= total_pages):
                interactions["page_changed"] = total_pages
        
        return interactions
    
    @staticmethod
    def _render_page_break(page_number: int):
        """Render page break visualization"""
        st.markdown(
            f'''
            <div class="page-break">
                <span class="page-break-label">Page {page_number} End</span>
            </div>
            ''',
            unsafe_allow_html=True
        )
    
    @staticmethod
    def split_content_into_pages(
        content: str,
        content_type: str = "html",
        page_size: PageSize = PageSize.A4,
        page_height_chars: int = 50,
        preserve_paragraphs: bool = True
    ) -> List[PageContent]:
        """
        Split content into pages for multi-page display.
        
        Args:
            content: Content to split
            content_type: Type of content
            page_size: Target page size
            page_height_chars: Approximate characters per page (rough estimate)
            preserve_paragraphs: Whether to preserve paragraph boundaries
            
        Returns:
            List of PageContent objects
        """
        pages = []
        
        if not content:
            return pages
        
        if preserve_paragraphs and content_type in ["html", "markdown"]:
            # Split by paragraphs for better page breaks
            if content_type == "html":
                paragraphs = content.split('</p>')
                paragraphs = [p + '</p>' for p in paragraphs[:-1]] + [paragraphs[-1]]
            else:  # markdown
                paragraphs = content.split('\n\n')
        else:
            # Simple character-based splitting
            chunk_size = page_height_chars * 80  # Rough estimate
            paragraphs = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        current_page_content = ""
        current_page_length = 0
        page_number = 1
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed page limit
            if current_page_length + len(paragraph) > page_height_chars * 80 and current_page_content:
                # Create new page
                pages.append(PageContent(
                    page_number=page_number,
                    content=current_page_content.strip(),
                    content_type=content_type,
                    page_size=page_size
                ))
                
                # Start new page
                current_page_content = paragraph
                current_page_length = len(paragraph)
                page_number += 1
            else:
                # Add to current page
                if current_page_content:
                    current_page_content += "\n\n" if content_type == "markdown" else ""
                current_page_content += paragraph
                current_page_length += len(paragraph)
        
        # Add final page if there's content
        if current_page_content.strip():
            pages.append(PageContent(
                page_number=page_number,
                content=current_page_content.strip(),
                content_type=content_type,
                page_size=page_size
            ))
        
        return pages
    
    @staticmethod
    def render_page_size_selector(
        current_size: PageSize,
        container_key: str = "page_size"
    ) -> Optional[PageSize]:
        """Render page size selector"""
        size_options = list(PageSize)
        size_names = [size.value["name"] for size in size_options]
        
        current_index = size_options.index(current_size)
        
        selected_index = st.selectbox(
            "Page Size",
            options=range(len(size_options)),
            index=current_index,
            format_func=lambda x: size_names[x],
            key=f"{container_key}_selector"
        )
        
        if selected_index != current_index:
            return size_options[selected_index]
        
        return None
    
    @staticmethod
    def render_display_mode_selector(
        current_mode: PageDisplayMode,
        container_key: str = "display_mode"
    ) -> Optional[PageDisplayMode]:
        """Render display mode selector"""
        mode_options = list(PageDisplayMode)
        mode_labels = {
            PageDisplayMode.SINGLE_PAGE: "Single Page",
            PageDisplayMode.CONTINUOUS: "Continuous",
            PageDisplayMode.FACING_PAGES: "Facing Pages",
            PageDisplayMode.THUMBNAIL_GRID: "Thumbnails"
        }
        
        current_index = mode_options.index(current_mode)
        
        selected_index = st.selectbox(
            "Display Mode",
            options=range(len(mode_options)),
            index=current_index,
            format_func=lambda x: mode_labels[mode_options[x]],
            key=f"{container_key}_selector"
        )
        
        if selected_index != current_index:
            return mode_options[selected_index]
        
        return None


# Utility functions

def create_page_from_content(
    content: str,
    page_number: int,
    content_type: str = "html",
    page_size: PageSize = PageSize.A4
) -> PageContent:
    """Create a PageContent object from content"""
    return PageContent(
        page_number=page_number,
        content=content,
        content_type=content_type,
        page_size=page_size
    )


def get_page_dimensions(page_size: PageSize, zoom_level: int = 100) -> Tuple[int, int]:
    """Get page dimensions in pixels for a given size and zoom"""
    size_info = page_size.value
    width = int(size_info["width"] * zoom_level / 100)
    height = int(size_info["height"] * zoom_level / 100)
    return width, height


def calculate_pages_needed(
    content_length: int,
    chars_per_page: int = 4000
) -> int:
    """Calculate approximate number of pages needed for content"""
    return max(1, math.ceil(content_length / chars_per_page))