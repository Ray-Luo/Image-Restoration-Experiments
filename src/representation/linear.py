class Linear:
    def __init__(self):
        pass

    def __call__(self, x):
        # [0.05, 4000] --> [0, 1]
        return x / 4000.0
