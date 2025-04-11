import re 

def format_queue_strings(input_filename):
    '''
    Read txt file of queues and convert to sql format
    '''
    # Read the content from the input file
    with open(input_filename, 'r') as file:
        data = file.read().splitlines()

    # Process the list of strings, remove any unwanted spaces or characters
    formatted_data = ["'" + line.strip().replace("\\", "") + "'\n" for line in data if line.strip()]  # Remove empty lines and strip spaces

    # Join the formatted strings with commas and wrap them in triple quotes
    formatted_output = '(' + ','.join(formatted_data) + ')'

    
    # Remove all backslashes except those before 'n' (i.e., not touching '\n')
    formatted_output = re.sub(r'\\(?!n)', '', formatted_output)

    return formatted_output

