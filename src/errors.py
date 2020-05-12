class Error(Exception):
    """
    класс для ошибок внутри моего кода и алгоритмов
    """

    def __init__(self, msg=None):
        self.message = f'Unknown error' if msg is None else msg
