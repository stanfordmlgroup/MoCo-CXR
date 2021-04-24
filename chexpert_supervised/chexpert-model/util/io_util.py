import argparse
from sys import stderr


def args_to_list(csv, allow_empty, arg_type=int, allow_negative=True):
    """Convert comma-separated arguments to a list.

    Args:
        csv: Comma-separated list of arguments as a string.
        allow_empty: If True, allow the list to be empty. Otherwise return None instead of empty list.
        arg_type: Argument type in the list.
        allow_negative: If True, allow negative inputs.

    Returns:
        List of arguments, converted to `arg_type`.
    """
    arg_vals = [arg_type(d) for d in str(csv).split(',')]
    if not allow_negative:
        arg_vals = [v for v in arg_vals if v >= 0]
    if not allow_empty and len(arg_vals) == 0:
        return None
    return arg_vals

# TODO: Move to logger
def print_err(*args, **kwargs):
    """Print a message to stderr."""
    print(*args, file=stderr, **kwargs)


def str_to_bool(arg):
    """Convert an argument string into its boolean value.

    Args:
        arg: String representing a bool.

    Returns:
        Boolean value for the string.
    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
