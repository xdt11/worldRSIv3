from pathlib import Path
_ROOT=Path(__file__).parent
MESSAGE_SUMMARIZER=(_ROOT/"message_summarizer.txt").read_text()
MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE=(_ROOT/"message_summarizer_bridge.txt").read_text()
MESSAGE_SUMMARIZER_TEXT_ONLY=(_ROOT/"message_summarizer_text_only.txt").read_text()
