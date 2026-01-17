"""
Pagination state management.

Handles page tracking for gallery display.
"""

import math


class PaginationState:
    """Manages pagination state for gallery views."""
    
    def __init__(self, items_per_page: int = 50):
        """
        Initialize pagination state with a given items-per-page setting.
        
        Parameters:
        	items_per_page (int): Number of items to display per page; defaults to 50. Initializes the current page to 1 and the total item count to 0.
        """
        self.current_page = 1
        self.items_per_page = items_per_page
        self._total_items = 0
    
    def set_total_items(self, count: int):
        """
        Set the total number of items and adjust the current page if it exceeds the new total.
        
        Parameters:
            count (int): Total number of items available; used to recompute total pages. If the current page becomes greater than the recomputed total pages, current_page is clamped to the highest available page (minimum 1).
        """
        self._total_items = count
        if self.current_page > self.total_pages:
            self.current_page = max(1, self.total_pages)
    
    @property
    def total_pages(self) -> int:
        """
        Compute the number of pages required to display all items given the current items_per_page.
        
        Returns:
            int: Total page count; returns 1 when there are zero items, otherwise the ceiling of (total items / max(1, items_per_page)).
        """
        if self._total_items == 0:
            return 1
        return math.ceil(self._total_items / max(1, self.items_per_page))
    
    def next_page(self) -> bool:
        """
        Advance to the next page if one exists.
        
        Returns:
            True if the current page was incremented, False otherwise.
        """
        if self.current_page < self.total_pages:
            self.current_page += 1
            return True
        return False
    
    def prev_page(self) -> bool:
        """
        Move to the previous page if possible.
        
        Returns:
            True if the current page was decremented, False otherwise.
        """
        if self.current_page > 1:
            self.current_page -= 1
            return True
        return False
    
    def jump_to_page(self, page_num: int) -> bool:
        """
        Set the current page to the specified page number if it is within the valid range.
        
        Parameters:
            page_num: A value convertible to int representing the desired page number.
        
        Returns:
            `True` if the page was set to `page_num`, `False` otherwise.
        """
        try:
            val = int(page_num)
            if 1 <= val <= self.total_pages:
                self.current_page = val
                return True
        except (ValueError, TypeError):
            pass
        return False
    
    def update_items_per_page(self, count: int):
        """
        Set items per page to a positive integer and reset the current page to 1.
        
        Parameters:
            count (int | any): Value to use for items per page; the function attempts to cast this to `int`. If the cast succeeds and the resulting value is greater than 0, `items_per_page` is updated and `current_page` is set to 1. If casting raises a `ValueError`, the method leaves state unchanged.
        """
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
        """
        Return the current page and total pages as a compact "current_page/total_pages" string.
        
        Returns:
            A string formatted as "current_page/total_pages" (for example, "2/5").
        """
        return f"{self.current_page}/{self.total_pages}"
    
    def get_total_label(self) -> str:
        """
        Produce a label showing the total number of pages, prefixed by "/ ".
        
        Returns:
            label (str): String in the format "/ N" where N is the total number of pages.
        """
        return f"/ {self.total_pages}"
    
    def is_visible(self) -> bool:
        """
        Determine whether pagination controls should be shown.
        
        Returns:
            bool: True if there is more than one page, False otherwise.
        """
        return self.total_pages > 1
    
    def get_slice(self) -> tuple:
        """
        Return the zero-based slice indices for the current page.
        
        The start index is inclusive and the end index is exclusive, suitable for slicing sequences.
        
        Returns:
            (start_index, end_index) (tuple): Start (inclusive) and end (exclusive) indices for the current page.
        """
        start = (self.current_page - 1) * self.items_per_page
        end = start + self.items_per_page
        return start, end