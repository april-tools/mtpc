import uuid
import datetime


def unique_timestamp():
    now = datetime.datetime.now()
    formatted = now.strftime("%Y-%m-%d-%H-%M-%S")
    unique_suffix = str(uuid.uuid4())[:8]
    return f"{formatted}-{unique_suffix}"
