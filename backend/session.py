class SessionState:
    def __init__(self):
        self.sequence: list = []
        self.hand_flags: list = []
        self.no_hand_streak: int = 0
        self.glossas: list = []
        self.last_prediction: str = ""
