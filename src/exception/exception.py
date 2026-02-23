def error_message_details(error: Exception):
    tb = error.__traceback__
    file_name = tb.tb_frame.f_code.co_filename
    line_number = tb.tb_lineno

    return (
        f"Error occured in python script : {file_name}, "
        f"Line number : {line_number}, "
        f"Error message : {error}"
    )

class CustomException(Exception):
    def __init__(self, error: Exception):
        super().__init__(error_message_details(error=error))