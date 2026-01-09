"""
Console Logging Utility

Centralizes print logic to support toggleable verbose output.
Allows printing in two modes:
1. Verbose (Default): Prints everything.
2. Minimal: Prints only critical information (start, stats, errors) via force=True.
"""

from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

class ConsoleLogger:
    def __init__(self):
        self.verbose = True
    
    def set_verbose(self, verbose: bool):
        """Set whether to print non-forced messages."""
        self.verbose = verbose
    
    def print(self, *args, force=False, color=None, end="\n", **kwargs):
        """
        Print message to console.
        
        Args:
            *args: Things to print
            force (bool): If True, print even if verbose is False.
            color (colorama.Fore): Color to wrap the message in.
            end (str): Line ending.
            **kwargs: Passed to built-in print.
        """
        if not self.verbose and not force:
            return
        
        msg = " ".join(str(a) for a in args)
        
        if color:
            msg = f"{color}{msg}{Style.RESET_ALL}"
            
        print(msg, end=end, **kwargs)
        
    def header(self, text, char="=", width=60, color=Fore.WHITE, force=False):
        """Print a centered header block."""
        if not self.verbose and not force:
            return
        
        border = char * width
        self.print(f"{color}{border}", force=True) # Recursive call handles color
        self.print(f"{color}{text.center(width)}", force=True)
        self.print(f"{color}{border}{Style.RESET_ALL}", force=True)
        
    def section(self, title, color=Fore.WHITE, force=False):
        """Print a section header."""
        if not self.verbose and not force:
            return
            
        self.print(f"\n{color}{'─' * 50}", force=True)
        self.print(f"{color}  {title}", force=True)
        self.print(f"{color}{'─' * 50}{Style.RESET_ALL}", force=True)
        
    def item(self, label, value="", color=None, force=False):
        """Print a labeled item."""
        if not self.verbose and not force:
            return
            
        c = color or Fore.WHITE
        if value:
            self.print(f"{c}  {label}:{Style.RESET_ALL} {value}", force=True)
        else:
            self.print(f"{c}  {label}{Style.RESET_ALL}", force=True)

    def error(self, msg, force=True):
        """Print error message (always forced by default)."""
        self.print(f"{Fore.RED}Error: {msg}{Style.RESET_ALL}", force=force)

    def warning(self, msg, force=True):
        """Print warning message."""
        self.print(f"{Fore.YELLOW}Warning: {msg}{Style.RESET_ALL}", force=force)
    
    def success(self, msg, force=False):
        """Print success message."""
        self.print(f"{Fore.GREEN}{msg}{Style.RESET_ALL}", force=force)


# Global instance
console = ConsoleLogger()
