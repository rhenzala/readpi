import picamera
import pygame
import io

# Initialize PiCamera
camera = picamera.PiCamera()
camera.resolution = (640, 480)

# Initialize pygame
pygame.init()

# Set display size
display_size = (640, 480)
screen = pygame.display.set_mode(display_size)
pygame.display.set_caption("Camera Feed Navigation")

# Initial ROI coordinates
roi_box = [0, 0, 320, 240]  # [left, upper, width, height]

try:
    running = True
    while running:
        # Capture image
        stream = io.BytesIO()
        camera.capture(stream, format='rgb', use_video_port=True)
        stream.seek(0)

        # Convert stream to Surface
        raw_frame = pygame.image.frombuffer(stream.getvalue(), (640, 480), 'RGB')

        # Crop frame to ROI
        cropped_frame = raw_frame.subsurface(roi_box)

        # Display frame
        screen.blit(cropped_frame, (0, 0))
        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Adjust ROI based on arrow key presses
                if event.key == pygame.K_UP:
                    roi_box[1] -= 10
                elif event.key == pygame.K_DOWN:
                    roi_box[1] += 10
                elif event.key == pygame.K_LEFT:
                    roi_box[0] -= 10
                elif event.key == pygame.K_RIGHT:
                    roi_box[0] += 10
                elif event.key == pygame.K_EQUALS:  # Zoom in
                    roi_box[2] -= 20
                    roi_box[3] -= 15
                elif event.key == pygame.K_MINUS:   # Zoom out
                    roi_box[2] += 20
                    roi_box[3] += 15

                # Ensure ROI stays within bounds of the camera resolution
                roi_box[0] = max(0, min(roi_box[0], 640 - roi_box[2]))
                roi_box[1] = max(0, min(roi_box[1], 480 - roi_box[3]))
                roi_box[2] = max(10, min(roi_box[2], 640 - roi_box[0]))
                roi_box[3] = max(10, min(roi_box[3], 480 - roi_box[1]))

finally:
    # Clean up resources
    camera.close()
    pygame.quit()
