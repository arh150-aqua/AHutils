import cv2
import os



def extract_frames_from_video(input_video_path, output_dir = '.'):
    '''
    Extracts frames from a video file and saves them as individual image files.

    Parameters:
        input_video_path (str): The path to the input video file.
        output_dir (str, optional): The directory where the extracted frames will be saved. Default is the current directory ('.').

    Raises:
        Exception: If the video file cannot be opened.
    
    Returns:
        None: The function saves the frames as image files in the specified directory.
    '''
    

    video_capture = cv2.VideoCapture(input_video_path)
    if not video_capture.isOpened():
        raise Exception(f"Could not open video file: {input_video_path}")

    fnum = 0
    while True:
        
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video
        
        image_name = os.path.join(output_dir, f'fram_{fnum}.png')
        print('saving frame: ', image_name)
        cv2.imwrite(image_name, frame)
        fnum += 1
    
    