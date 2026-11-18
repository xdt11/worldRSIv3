"""Execute commands only in the supplied task sandbox."""
def bash(ctx, box, command, timeout_s=120, max_output_chars=None):
    # Feedback allowance is consumed by the role adapter, not the shell runner.
    return box.bash(command=command, timeout_s=timeout_s), None
