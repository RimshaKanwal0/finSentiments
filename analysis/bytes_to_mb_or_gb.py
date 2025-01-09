
# Function to convert bytes to MB or GB
def bytes_to_mb_or_gb(byte_value):
    if byte_value >= 1024**3:  # GB
        return f"{byte_value / (1024**3):.2f} GB"
    elif byte_value >= 1024**2:  # MB
        return f"{byte_value / (1024**2):.2f} MB"
    else:  # bytes
        return f"{byte_value} bytes"