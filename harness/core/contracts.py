"""Project diagnostics shared by runtime and tools."""


class ToolError(ValueError):
    """Actionable tool diagnostics; adapters may supply a stable failure signature."""
    def __init__(self, code, message, **details):
        super().__init__(message)
        self.diagnostic = {'code':code, 'message':message, **details}


class EnvironmentChanged(ValueError):
    def __init__(self, message):
        super().__init__(message)
        self.reason = {'code':'harness_changed','message':message}
