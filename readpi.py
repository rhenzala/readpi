#!/usr/bin/python3

import cv2
import time
import pygame
import pytesseract
import threading
import pyaudio
import os                      
import re                      
import signal                  
import subprocess              
import sys
import numpy as np
from imutils.perspective import four_point_transform
from vosk import Model, KaldiRecognizer
from picamera import PiCamera
from picamera.array import PiRGBArray
from gpiozero import Button

# importing the readpi python files
import audio


model = Model(r"/home/readpi/Downloads/vosk-model-small-en-us-0.15")
recognizer = KaldiRecognizer(model, 16000)

mic = pyaudio.PyAudio()

def speak(ai):
    subprocess.call(['pico2wave', '-w', 'temp.wav', ai])
    # Use sox to adjust the speed of the generated audio
    subprocess.call(['sox', 'temp.wav', 'temp_adjusted.wav', 'speed', '0.92'])
    # Play the adjusted audio using aplay
    subprocess.call(['aplay', 'temp_adjusted.wav'])


engine = "picotts"

# Parameters to set which functionality to run at a given time
listening = False
active_mode = False
read_out = False
magnification = False
shutdown = False
adjust_volume_up = False
adjust_volume_down = False


# define GPIO pins for (optional) push buttons
PIN_NUMBER_MAGNIFY =  4 # physical 7, scale button
PIN_NUMBER_ZOOMIN = 23 # physical 16
PIN_NUMBER_ZOOMOUT = 24 # physical 18
PIN_NUMBER_SHUTDOWN = 26 # pysical 37, shutdown button
PIN_NUMBER_VOLUMEUP = 27 # physical 13
PIN_NUMBER_VOLUMEDOWN = 17 # physical 11
PIN_NUMBER_TTS = 22 # physical 15
PIN_NUMBER_STOP = 25 # physical 22
PIN_NUMBER_UP = 5 # physical 29
PIN_NUMBER_DOWN = 6 # physical 31
PIN_NUMBER_LEFT = 13 # physical 33
PIN_NUMBER_RIGHT = 19 # physical 35

BTN_MAGNIFY = Button(PIN_NUMBER_MAGNIFY)
BTN_ZOOMIN = Button(PIN_NUMBER_ZOOMIN)
BTN_ZOOMOUT = Button(PIN_NUMBER_ZOOMOUT)
BTN_TTS = Button(PIN_NUMBER_TTS)
BTN_VOLUP = Button(PIN_NUMBER_VOLUMEUP)
BTN_VOLDOWN = Button(PIN_NUMBER_VOLUMEDOWN)
BTN_SHUTDOWN = Button(PIN_NUMBER_SHUTDOWN)
BTN_STOP = Button(PIN_NUMBER_STOP)
BTN_UP = Button(PIN_NUMBER_UP)
BTN_DOWN = Button(PIN_NUMBER_DOWN)
BTN_LEFT = Button(PIN_NUMBER_LEFT)
BTN_RIGHT = Button(PIN_NUMBER_RIGHT)

# used by the save_image(), modified inside init_camera hence defined as global. Also used by voice command outside the init_camera.
warped = None
image = None

def get_command():
    listening = True
    stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
    while listening:
        stream.start_stream()
        try:
            data = stream.read(4096)
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                response = result[14:-3]
                listening = False
                stream.close()
                return response
        except OSError:
            pass

def magnify():
    # Window size
    WINDOW_WIDTH    = 1024
    WINDOW_HEIGHT   = 1024
    WINDOW_SURFACE  = pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE
    image_filename  = None

    PAN_BOX_WIDTH   = 64
    PAN_BOX_HEIGHT  = 64
    PAN_STEP        = 5

    image_filename = "preview.jpg"
    ### PyGame initialisation
    pygame.init()
    window = pygame.display.set_mode( ( WINDOW_WIDTH, WINDOW_HEIGHT ), WINDOW_SURFACE )
    pygame.display.set_caption("Image Pan")

    ### Can we load the user's image OK?
    try:
        base_image = pygame.image.load( image_filename ).convert()
    except:
        errorExit( "Failed to open [%s]" % ( image_filename ) )

    ### Pan-position
    background = pygame.Surface( ( WINDOW_WIDTH, WINDOW_HEIGHT ) )   # zoomed section is copied here
    zoom_image = None
    pan_box    = pygame.Rect( 0, 0, PAN_BOX_WIDTH, PAN_BOX_HEIGHT )  # curent pan "cursor position"
    last_box   = pygame.Rect( 0, 0, 1, 1 )

    ### Main Loop
    clock = pygame.time.Clock()
    done = False
    while not done:

        # Handle user-input
        for event in pygame.event.get():
            if ( event.type == pygame.QUIT ):
                done = True

        # Movement keys
        # Pan-box moves up/down/left/right, Zooms with + and -
        keys = pygame.key.get_pressed()
        if ( BTN_STOP.is_pressed ):
            done = True
        if ( keys[pygame.K_UP] or BTN_UP.is_pressed ):
            pan_box.y -= PAN_STEP
        if ( keys[pygame.K_DOWN] or BTN_DOWN.is_pressed ):
             pan_box.y += PAN_STEP
        if ( keys[pygame.K_LEFT] or BTN_LEFT.is_pressed ):
            pan_box.x -= PAN_STEP
        if ( keys[pygame.K_RIGHT] or BTN_RIGHT.is_pressed ):
            pan_box.x += PAN_STEP
        if ( keys[pygame.K_PLUS] or keys[pygame.K_EQUALS] or BTN_ZOOMOUT.is_pressed ):
            pan_box.width  += PAN_STEP
            pan_box.height += PAN_STEP
            if ( pan_box.width > WINDOW_WIDTH ):  # Ensure size is sane
                pan_box.width = WINDOW_WIDTH
            if ( pan_box.height > WINDOW_HEIGHT ):
                pan_box.height = WINDOW_HEIGHT
        if ( keys[pygame.K_MINUS] or BTN_ZOOMIN.is_pressed ):
            pan_box.width  -= PAN_STEP
            pan_box.height -= PAN_STEP
            if ( pan_box.width < PAN_STEP ):  # Ensure size is sane
                pan_box.width = PAN_STEP
            if ( pan_box.height < PAN_STEP ):
                pan_box.height = PAN_STEP

        # Ensure the pan-box stays within image
        PAN_BOX_WIDTH  = min( PAN_BOX_WIDTH, base_image.get_width() )
        PAN_BOX_HEIGHT = min( PAN_BOX_HEIGHT, base_image.get_height() )
        if ( pan_box.x < 0 ):
            pan_box.x = 0 
        elif ( pan_box.x + pan_box.width >= base_image.get_width() ):
            pan_box.x = base_image.get_width() - pan_box.width - 1
        if ( pan_box.y < 0 ):
            pan_box.y = 0 
        elif ( pan_box.y + pan_box.height >= base_image.get_height() ):
            pan_box.y = base_image.get_height() - pan_box.height - 1

        # Re-do the zoom, but only if the pan box has changed since last time
        if ( pan_box != last_box ):
            # Create a new sub-image but only if the size changed
            # otherwise we can just re-use it
            if ( pan_box.width != last_box.width or pan_box.height != last_box.height ):
                zoom_image = pygame.Surface( ( pan_box.width, pan_box.height ) )  
            
            zoom_image.blit( base_image, ( 0, 0 ), pan_box )                  # copy base image
            window_size = ( WINDOW_WIDTH, WINDOW_HEIGHT )
            pygame.transform.scale( zoom_image, window_size, background )     # scale into thebackground
            last_box = pan_box.copy()                                         # copy current position

        window.blit( background, ( 0, 0 ) )
        pygame.display.flip()

        # Clamp FPS
        clock.tick_busy_loop(60)

    pygame.quit()

def readout():
    file_name = "preview.jpg"
    ocr_result = pytesseract.image_to_string(file_name)
    print(ocr_result)
    
    # Define a function to speak asynchronously
    def speak_async(text):
        subprocess.call(['pico2wave', '-w', 'temp.wav', text])
        subprocess.call(['sox', 'temp.wav', 'temp_adjusted.wav', 'speed', '0.92'])
        subprocess.call(['aplay', 'temp_adjusted.wav'])
    
    # Start a new thread for speaking
    speak_thread = threading.Thread(target=speak_async, args=(ocr_result,))
    speak_thread.start()
        
def power_off():
    print("Powering off...")
    subprocess.run(["sudo", "poweroff"])
    
def stop_playing():
    # Use pgrep to find the PID of 'aplay' process
    process = subprocess.Popen(['pgrep', '-f', 'aplay'], stdout=subprocess.PIPE)
    output, _ = process.communicate()
    process_ids = output.split()

    if process_ids:
        for pid in process_ids:
            # Kill the process using the PID
            subprocess.run(['kill', pid.decode('utf-8')])

        print("Stopped 'aplay' process successfully.")
    else:
        print("No 'aplay' process found.")
    
    
def save_image(image):
    cv2.imwrite('preview.jpg', image)
    print("Image saved as 'preview.jpg'")


def init_camera():
    global rawCapture
    
    WIDTH, HEIGHT = 512, 512
    # Initialize the camera and grab a reference to the raw capture
    camera = PiCamera()
    camera.resolution = (WIDTH, HEIGHT)  # You can adjust the resolution here
    camera.framerate = 30
    camera.rotation = 90
    rawCapture = PiRGBArray(camera, size=(WIDTH, HEIGHT))
    
    # Allow the camera to warmup
    time.sleep(0.1)
    
    # Function to continuously capture frames from the camera
    def capture_frames():
        global rawCapture
        global image
        global warped
        

        def scan_detection(image):
            global document_contour
            document_contour = np.array([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            max_area = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:
                    peri = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
                    if area > max_area and len(approx) == 4:
                        document_contour = approx
                        max_area = area

            cv2.drawContours(image, [document_contour], -1, (0, 255, 0), 3)
            
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            # Grab the raw NumPy array representing the image
            image = frame.array

            # Display the frame (you might need to adjust this part)
            #cv2.imshow("Camera Feed", image)
            frame_copy = image.copy()
    
            scan_detection(frame_copy)
            
            #cv2.imshow("input", image)
            
            warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
            #cv2.imshow("Warped", warped)
            
        
            cv2.imshow("Warped", warped)

            # Wait for key press (press 'q' to quit)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") or BTN_STOP.is_pressed:
                stop_playing()
            elif key == ord("m") or BTN_MAGNIFY.is_pressed:
                if warped is not None:
                    save_image(image)
                    time.sleep(0.1)
                    magnify()
                else:
                    print("Processed image is empty. Cannot save.")
            elif key == ord("r") or BTN_TTS.is_pressed:
                if warped is not None:
                    save_image(image)
                    time.sleep(0.1)
                    readout()
                else:
                    print("Processed image is empty. Cannot save.")
            elif key == ord("o") or BTN_VOLUP.is_pressed:
                audio.volume_up()
            elif key == ord("p") or BTN_VOLDOWN.is_pressed:
                audio.volume_down()
            elif key == 27 or BTN_SHUTDOWN.is_pressed: # esc key to poweroff
                power_off()

            # Clear the stream for the next frame
            rawCapture.truncate(0)
    
    # Start capturing frames in a separate thread
    camera_thread = threading.Thread(target=capture_frames)
    camera_thread.daemon = True  # Daemonize the thread so it automatically exits when the main program ends
    camera_thread.start()
    
    
camera = init_camera()

# Voice command constantly listens for commands, wake word is either "hello" or "computer"
while True:
    print("Waiting for command...")
    command = get_command()
    if command == "":
        pass
    elif command == "hello" or command == "computer":
        active_mode = True
        speak("Hello user")
        
    if active_mode:
        command = get_command()
        if command == "say something":
            speak("nice")
        elif command == "what time is it":
            current_time = subprocess.check_output(['date']).decode('utf-8').strip()
            speak(f"The time is {current_time}")
        elif command == "volume down":
            speak("Okay. Decreasing the volume.")
            adjust_volume_down = True
            active_mode = False
        elif command == "volume up":
            speak("Okay. Increasing the volume.")
            adjust_volume_up = True
            active_mode = False
        elif command == "read out" or command == "text to speech":
            speak("Okay. Text to speech mode.")
            read_out = True
            active_mode = False
        elif command == "magnify":
            speak("Okay. Magnification mode.")
            magnification = True
            active_mode = False
        elif command == "power off" or command == "shutdown":
            speak("Okay. Shutting down.")
            shutdown = True
            active_mode = False
        elif command == "goodbye" or command == "exit":
            speak("Thank you. Terminating voice command")
            break
        else:
            speak("Command not recognized")
            
    # activate and run each functinality and then set it to inactive again to avoid conflict
    if read_out:
        if warped is not None:
            save_image(image)
            time.sleep(0.1)
            readout()
            read_out = False  
        else:
            print("Processed image is empty. Cannot save.")
    if magnification:
        if warped is not None:
            save_image(image)
            time.sleep(0.1)
            magnify()
            magnification = False  
        else:
            print("Processed image is empty. Cannot save.")
    if shutdown:
        power_off()
        shutdown = False
    if adjust_volume_up:
        audio.volume_up()
        adjust_volume_up = False
    if adjust_volume_down:
        audio.volume_down()
        adjust_volume_down = False



