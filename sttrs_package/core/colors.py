"""Nord theme colors and terminal styling utilities."""

# Nord theme colors (Snow Storm, Frost, and Aurora)
FOREGROUND = "\033[38;2;216;222;233m"  # Snow Storm #d8dee9 - Default text
TEXT_LIGHT = "\033[38;2;229;233;240m"  # Snow Storm #e5e9f0 - Light text
TEXT_LIGHTER = "\033[38;2;236;239;244m"  # Snow Storm #eceff4 - Lighter text

# Frost colors for cool, calming elements
SAGE = "\033[38;2;143;188;187m"  # Frost #8fbcbb - Sage accent
FROST = "\033[38;2;136;192;208m"  # Frost #88c0d0 - Ice blue
STEEL = "\033[38;2;129;161;193m"  # Frost #81a1c1 - Steel blue

# Aurora colors for highlights and status
RED = "\033[38;2;191;97;106m"     # Aurora #bf616a - Error/warning
ORANGE = "\033[38;2;208;135;112m"  # Aurora #d08770 - Caution
YELLOW = "\033[38;2;235;203;139m"  # Aurora #ebcb8b - Important items
GREEN = "\033[38;2;163;190;140m"   # Aurora #a3be8c - Success
PURPLE = "\033[38;2;180;142;173m"  # Aurora #b48ead - Headers/special

# Style reset
RESET = "\033[0m"

def apply_style(text: str, *styles: str) -> str:
    """Apply one or more styles to text."""
    return "".join(styles) + text + RESET

def init_terminal():
    """Initialize terminal with default styling."""
    pass  # No background color initialization

def clear_screen():
    """Clear the screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear') 