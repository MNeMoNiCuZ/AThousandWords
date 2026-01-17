"""
Pagination state management.

Handles page tracking for gallery display.
"""

import math


class PaginationState:
    """Manages pagination state for gallery views."""
    
    def __init__(self, items_per_page: int = 50):
        self.current_page = 1
        self.items_per_page = items_per_page
        self._total_items = 0
    
    def set_total_items(self, count: int):
        """Update total items count."""
        self._total_items = count
        if self.current_page > self.total_pages:
            self.current_page = max(1, self.total_pages)
    
    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        if self._total_items == 0:
            return 1
        return math.ceil(self._total_items / max(1, self.items_per_page))
    
    def next_page(self) -> bool:
        """Go to next page. Returns True if page changed."""
        if self.current_page < self.total_pages:
            self.current_page += 1
            return True
        return False
    
    def prev_page(self) -> bool:
        """Go to previous page. Returns True if page changed."""
        if self.current_page > 1:
            self.current_page -= 1
            return True
        return False
    
    def jump_to_page(self, page_num: int) -> bool:
        """Jump to specific page. Returns True if valid."""
        try:
            val = int(page_num)
            if 1 <= val <= self.total_pages:
                self.current_page = val
                return True
        except (ValueError, TypeError):
            pass
        return False
    
    def update_items_per_page(self, count: int):
        """Update items per page and reset to page 1."""
        try:
            val = int(count)
            if val > 0:
                self.items_per_page = val
                self.current_page = 1
        except ValueError:
            pass
    
    def reset(self):
        """Reset to page 1."""
        self.current_page = 1
    
    def get_page_info(self) -> str:
        """Get pagination info string."""
        return f"{self.current_page}/{self.total_pages}"
    
    def get_total_label(self) -> str:
        """Get total pages label string."""
        return f"/ {self.total_pages}"
    
    def is_visible(self) -> bool:
        """Should pagination controls be visible?"""
        return self.total_pages > 1
    
    def get_slice(self) -> tuple:
        """Get slice indices for current page.
        
        Returns (start_index, end_index).
        """
        start = (self.current_page - 1) * self.items_per_page
        end = start + self.items_per_page
        return start, end
