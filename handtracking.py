import os
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the Hands module
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Load the video file
    video_path = r'C:\backup\Notes\Design\hand tracking\media\sesting2.mp4'
    cap = cv2.VideoCapture(video_path)

    # Get the video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Construct the output video path in the same directory as the input video
    output_dir = os.path.dirname(video_path)
    output_filename = 'output_' + os.path.basename(video_path)
    output_path = os.path.join(output_dir, output_filename)

    # Create a VideoWriter object to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Convert the image to RGB and process it with MediaPipe Hands
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Extract hand landmark positions and display them on the screen
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the landmark positions for each hand
                landmark_positions = []
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                    landmark_positions.append((x, y))

                # Display the landmark positions on the screen
                for i, pos in enumerate(landmark_positions):
                    cv2.putText(image, str(i), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw the hand landmarks on the image
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Write the processed frame to the output video
        out.write(image)

        # Display the image
        cv2.imshow('Hand Tracking', image)

        # Exit on 'q' key press or when the video ends
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
