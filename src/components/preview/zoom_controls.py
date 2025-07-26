"""
Zoom Controls Component

Provides smooth scaling functionality for document previews with
preset zoom levels, slider controls, and keyboard shortcuts.
"""

import streamlit as st
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ZoomLevel(Enum):
    """Predefined zoom levels"""
    ZOOM_25 = 25
    ZOOM_50 = 50
    ZOOM_75 = 75
    ZOOM_100 = 100
    ZOOM_125 = 125
    ZOOM_150 = 150
    ZOOM_175 = 175
    ZOOM_200 = 200
    ZOOM_250 = 250
    ZOOM_300 = 300
    ZOOM_400 = 400


@dataclass
class ZoomControlsConfig:
    """Configuration for zoom controls"""
    min_zoom: int = 25
    max_zoom: int = 400
    zoom_step: int = 25
    show_slider: bool = True
    show_presets: bool = True
    show_percentage: bool = True
    enable_keyboard_shortcuts: bool = True
    preset_levels: List[int] = None
    
    def __post_init__(self):
        if self.preset_levels is None:
            self.preset_levels = [50, 75, 100, 125, 150, 200]


class ZoomControls:
    """
    Zoom controls component providing smooth scaling functionality.
    
    Offers multiple ways to control zoom: buttons, slider, presets,
    and keyboard shortcuts for optimal user experience.
    """
    
    @staticmethod
    def render_zoom_controls(
        current_zoom: int,
        config: Optional[ZoomControlsConfig] = None,
        on_zoom_change: Optional[Callable[[int], None]] = None,
        container_key: str = "zoom_controls"
    ) -> Dict[str, Any]:
        """
        Render complete zoom controls interface.
        
        Args:
            current_zoom: Current zoom level percentage
            config: Zoom controls configuration
            on_zoom_change: Callback for zoom changes
            container_key: Unique key for the container
            
        Returns:
            Dictionary with zoom interactions and state
        """
        if config is None:
            config = ZoomControlsConfig()
        
        interactions = {}
        
        # Create zoom controls container
        with st.container():
            st.markdown('<div class="zoom-controls">', unsafe_allow_html=True)
            
            # Layout columns for different control types
            col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 2])
            
            # Zoom out button
            with col1:
                zoom_out_interactions = ZoomControls._render_zoom_button(
                    "out", current_zoom, config, container_key
                )
                interactions.update(zoom_out_interactions)
            
            # Zoom in button
            with col2:
                zoom_in_interactions = ZoomControls._render_zoom_button(
                    "in", current_zoom, config, container_key
                )
                interactions.update(zoom_in_interactions)
            
            # Zoom slider
            if config.show_slider:
                with col3:
                    slider_interactions = ZoomControls._render_zoom_slider(
                        current_zoom, config, container_key
                    )
                    interactions.update(slider_interactions)
            
            # Zoom percentage display
            if config.show_percentage:
                with col4:
                    ZoomControls._render_zoom_percentage(current_zoom)
            
            # Zoom presets
            if config.show_presets:
                with col5:
                    preset_interactions = ZoomControls._render_zoom_presets(
                        current_zoom, config, container_key
                    )
                    interactions.update(preset_interactions)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Keyboard shortcuts info
            if config.enable_keyboard_shortcuts:
                ZoomControls._render_keyboard_shortcuts()
        
        # Call callback if zoom changed
        if interactions.get("zoom_changed") and on_zoom_change:
            on_zoom_change(interactions["zoom_changed"])
        
        return interactions
    
    @staticmethod
    def _render_zoom_button(
        button_type: str,
        current_zoom: int,
        config: ZoomControlsConfig,
        container_key: str
    ) -> Dict[str, Any]:
        """Render zoom in/out buttons"""
        interactions = {}
        
        if button_type == "out":
            icon = "🔍-"
            help_text = "Zoom Out (Ctrl+-)"
            disabled = current_zoom <= config.min_zoom
            new_zoom = max(config.min_zoom, current_zoom - config.zoom_step)
            button_key = f"{container_key}_zoom_out"
        else:
            icon = "🔍+"
            help_text = "Zoom In (Ctrl++)"
            disabled = current_zoom >= config.max_zoom
            new_zoom = min(config.max_zoom, current_zoom + config.zoom_step)
            button_key = f"{container_key}_zoom_in"
        
        if st.button(
            icon,
            key=button_key,
            help=help_text,
            disabled=disabled
        ):
            interactions["zoom_changed"] = new_zoom
            interactions["zoom_source"] = f"button_{button_type}"
        
        return interactions
    
    @staticmethod
    def _render_zoom_slider(
        current_zoom: int,
        config: ZoomControlsConfig,
        container_key: str
    ) -> Dict[str, Any]:
        """Render zoom slider control"""
        interactions = {}
        
        new_zoom = st.slider(
            "Zoom Level",
            min_value=config.min_zoom,
            max_value=config.max_zoom,
            value=current_zoom,
            step=config.zoom_step,
            key=f"{container_key}_zoom_slider",
            label_visibility="collapsed",
            help="Adjust zoom level"
        )
        
        if new_zoom != current_zoom:
            interactions["zoom_changed"] = new_zoom
            interactions["zoom_source"] = "slider"
        
        return interactions
    
    @staticmethod
    def _render_zoom_percentage(current_zoom: int):
        """Render zoom percentage display"""
        st.markdown(
            f'<div class="zoom-percentage">{current_zoom}%</div>',
            unsafe_allow_html=True
        )
    
    @staticmethod
    def _render_zoom_presets(
        current_zoom: int,
        config: ZoomControlsConfig,
        container_key: str
    ) -> Dict[str, Any]:
        """Render zoom preset buttons"""
        interactions = {}
        
        # Create columns for preset buttons
        preset_cols = st.columns(len(config.preset_levels))
        
        for i, preset_level in enumerate(config.preset_levels):
            with preset_cols[i]:
                # Determine button style based on current zoom
                if preset_level == current_zoom:
                    button_class = "zoom-preset active"
                else:
                    button_class = "zoom-preset"
                
                # Format preset label
                if preset_level == 100:
                    label = "Fit"
                else:
                    label = f"{preset_level}%"
                
                if st.button(
                    label,
                    key=f"{container_key}_preset_{preset_level}",
                    help=f"Set zoom to {preset_level}%"
                ):
                    interactions["zoom_changed"] = preset_level
                    interactions["zoom_source"] = f"preset_{preset_level}"
        
        return interactions
    
    @staticmethod
    def _render_keyboard_shortcuts():
        """Render keyboard shortcuts information"""
        with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
            st.markdown("""
            **Zoom Controls:**
            - `Ctrl` + `+` : Zoom In
            - `Ctrl` + `-` : Zoom Out  
            - `Ctrl` + `0` : Reset to 100%
            - `Ctrl` + `1` : Fit to Width
            - `Ctrl` + `2` : Fit to Page
            
            **Navigation:**
            - `Page Up/Down` : Scroll through document
            - `Home/End` : Go to beginning/end
            - `Ctrl` + `Home` : Go to first page
            - `Ctrl` + `End` : Go to last page
            """)
    
    @staticmethod
    def render_compact_zoom_controls(
        current_zoom: int,
        min_zoom: int = 25,
        max_zoom: int = 400,
        zoom_step: int = 25,
        container_key: str = "compact_zoom"
    ) -> Optional[int]:
        """
        Render compact zoom controls for limited space.
        
        Args:
            current_zoom: Current zoom level
            min_zoom: Minimum zoom level
            max_zoom: Maximum zoom level
            zoom_step: Zoom step size
            container_key: Unique key for the container
            
        Returns:
            New zoom level if changed, None otherwise
        """
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        
        with col1:
            if st.button("➖", key=f"{container_key}_out", disabled=current_zoom <= min_zoom):
                return max(min_zoom, current_zoom - zoom_step)
        
        with col2:
            new_zoom = st.select_slider(
                "Zoom",
                options=[25, 50, 75, 100, 125, 150, 200],
                value=current_zoom,
                key=f"{container_key}_select",
                label_visibility="collapsed"
            )
            if new_zoom != current_zoom:
                return new_zoom
        
        with col3:
            if st.button("➕", key=f"{container_key}_in", disabled=current_zoom >= max_zoom):
                return min(max_zoom, current_zoom + zoom_step)
        
        with col4:
            st.markdown(f"**{current_zoom}%**")
        
        return None
    
    @staticmethod
    def calculate_fit_to_width_zoom(
        document_width: int,
        container_width: int,
        margin: int = 40
    ) -> int:
        """
        Calculate zoom level to fit document width to container.
        
        Args:
            document_width: Document width in pixels
            container_width: Container width in pixels
            margin: Margin to leave on sides
            
        Returns:
            Calculated zoom percentage
        """
        available_width = container_width - (margin * 2)
        zoom_factor = available_width / document_width
        zoom_percentage = int(zoom_factor * 100)
        
        # Clamp to reasonable bounds
        return max(25, min(400, zoom_percentage))
    
    @staticmethod
    def calculate_fit_to_page_zoom(
        document_width: int,
        document_height: int,
        container_width: int,
        container_height: int,
        margin: int = 40
    ) -> int:
        """
        Calculate zoom level to fit entire document page.
        
        Args:
            document_width: Document width in pixels
            document_height: Document height in pixels
            container_width: Container width in pixels
            container_height: Container height in pixels
            margin: Margin to leave around document
            
        Returns:
            Calculated zoom percentage
        """
        available_width = container_width - (margin * 2)
        available_height = container_height - (margin * 2)
        
        width_zoom = available_width / document_width
        height_zoom = available_height / document_height
        
        # Use the smaller zoom factor to ensure entire page fits
        zoom_factor = min(width_zoom, height_zoom)
        zoom_percentage = int(zoom_factor * 100)
        
        # Clamp to reasonable bounds
        return max(25, min(400, zoom_percentage))
    
    @staticmethod
    def get_zoom_css_transform(zoom_level: int) -> str:
        """
        Get CSS transform string for zoom level.
        
        Args:
            zoom_level: Zoom percentage (25-400)
            
        Returns:
            CSS transform string
        """
        scale_factor = zoom_level / 100
        return f"transform: scale({scale_factor}); transform-origin: top center;"
    
    @staticmethod
    def handle_keyboard_zoom_shortcuts(
        current_zoom: int,
        config: Optional[ZoomControlsConfig] = None
    ) -> Optional[int]:
        """
        Handle keyboard shortcuts for zoom control.
        
        Note: This is a placeholder for keyboard event handling.
        Streamlit doesn't have direct keyboard event support,
        but this method provides the logic for when it's available.
        
        Args:
            current_zoom: Current zoom level
            config: Zoom configuration
            
        Returns:
            New zoom level if shortcut was used, None otherwise
        """
        if config is None:
            config = ZoomControlsConfig()
        
        # This would be implemented with JavaScript in a real application
        # For now, return None as Streamlit doesn't support keyboard events
        logger.info("Keyboard shortcuts not yet implemented in Streamlit")
        return None


# Utility functions for zoom calculations

def get_zoom_levels_for_range(min_zoom: int, max_zoom: int, step: int) -> List[int]:
    """Get list of zoom levels for a given range"""
    return list(range(min_zoom, max_zoom + 1, step))


def snap_to_nearest_zoom_level(
    zoom: int,
    available_levels: List[int]
) -> int:
    """Snap zoom value to nearest available level"""
    if not available_levels:
        return zoom
    
    return min(available_levels, key=lambda x: abs(x - zoom))


def is_zoom_level_valid(zoom: int, min_zoom: int = 25, max_zoom: int = 400) -> bool:
    """Check if zoom level is within valid range"""
    return min_zoom <= zoom <= max_zoom