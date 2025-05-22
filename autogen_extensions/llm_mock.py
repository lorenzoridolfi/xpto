class LLMMock:
    def __init__(self, static_response=None):
        self.static_response = (
            static_response or "DESCRIÇÃO: Mock\nRESUMO: Mock summary"
        )
        self.calls = []

    def create(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        class Choice:
            def __init__(self, content):
                self.message = type("Msg", (), {"content": content})()

        class Response:
            def __init__(self, content):
                self.choices = [Choice(content)]

        return Response(self.static_response)
