import unittest
from types import SimpleNamespace

from attention_maps.inference.comparison import DecodingConfig
from attention_maps.inference.gemini_translation import GeminiTranslationBackend


class FakeChat:
    def __init__(self):
        self.prompt = None

    def send_message(self, prompt):
        self.prompt = prompt
        return SimpleNamespace(text="नेपाली अनुवाद")


class FakeChats:
    def __init__(self):
        self.arguments = None
        self.chat = FakeChat()

    def create(self, **kwargs):
        self.arguments = kwargs
        return self.chat


class GeminiTranslationTests(unittest.TestCase):
    def test_uses_lima_pipeline_chat_generation_style(self):
        backend = object.__new__(GeminiTranslationBackend)
        backend.model_id = "gemini-3.5-flash-lite"
        backend.label = "lima-teacher:gemini-3.5-flash-lite"
        backend._client = SimpleNamespace(chats=FakeChats())

        output = backend.generate(
            "Translate this",
            DecodingConfig(
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                max_new_tokens=256,
            ),
        )

        self.assertEqual(output, "नेपाली अनुवाद")
        self.assertEqual(
            backend._client.chats.arguments["model"], "gemini-3.5-flash-lite"
        )
        config = backend._client.chats.arguments["config"]
        self.assertEqual(config.temperature, 0.2)
        self.assertEqual(config.top_k, 40)
        self.assertEqual(backend._client.chats.chat.prompt, "Translate this")


if __name__ == "__main__":
    unittest.main()
