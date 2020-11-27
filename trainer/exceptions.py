class ParamsError(Exception):
    """Exception raised for Incorrect Paramters.

    Attributes:
        expression: input expression in which the error occurred
        message: explanation of the error

    """

    def __init__(self, params_type, message):
        self.params_type = params_type
        self.message = message
