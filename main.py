import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize game variables
basket_width = 120
basket_height = 60
word_list = ["apple", "banana", "pear", "pineapple"]
words = []
score = 0

# Load basket image and resize it
basket_img = cv2.imread('Game_1/basket_1.png', cv2.IMREAD_UNCHANGED)  # Load with alpha channel if present
basket_img = cv2.resize(basket_img, (basket_width, basket_height))

# Function to create a new word
def create_word():
    x = random.randint(0, 640 - 100)  # Assuming max word width is 100 pixels
    return {'word': random.choice(word_list), 'x': x, 'y': 0}

# Function to overlay the basket image onto the frame
# Function to overlay the basket image onto the frame
def overlay_basket(frame, basket_img, x, y):
    # Define the region of interest (ROI) based on the basket's position
    y1, y2 = y, y + basket_img.shape[0]
    x1, x2 = x, x + basket_img.shape[1]

    # Ensure that the ROI is within frame boundaries
    if y1 < 0 or y2 > frame.shape[0] or x1 < 0 or x2 > frame.shape[1]:
        return  # Skip if the basket is out of bounds

    # Extract the basket region of interest (ROI)
    basket_roi = frame[y1:y2, x1:x2]

    # Check if the basket image has an alpha channel (transparency)
    if basket_img.shape[2] == 4:
        # Split the basket image into color channels and alpha channel
        basket_rgb = basket_img[:, :, :3]  # RGB channels
        mask = basket_img[:, :, 3]         # Alpha channel as mask

        # Convert the mask to a 3-channel mask to match the ROI
        mask_inv = cv2.bitwise_not(mask)   # Invert the mask
        mask_rgb = cv2.merge([mask] * 3)   # Make the mask 3-channel

        # Resize the mask to match the ROI size (if needed)
        if basket_roi.shape[:2] != mask_rgb.shape[:2]:
            mask_rgb = cv2.resize(mask_rgb, (basket_roi.shape[1], basket_roi.shape[0]))
            mask_inv = cv2.resize(mask_inv, (basket_roi.shape[1], basket_roi.shape[0]))

        # Black-out the area of the basket in the ROI (basket background)
        basket_bg = cv2.bitwise_and(basket_roi, basket_roi, mask=mask_inv)

        # Take only the region of the basket from the basket image
        basket_fg = cv2.bitwise_and(basket_rgb, basket_rgb, mask=mask)

        # Combine background and foreground
        result = cv2.add(basket_bg, basket_fg)
        frame[y1:y2, x1:x2] = result

    else:
        # If no alpha channel, just overlay the image
        frame[y1:y2, x1:x2] = basket_img


# Main game loop
cap = cv2.VideoCapture(1)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hand landmarks
    results = hands.process(rgb_frame)

    basket_x, basket_y = None, None  # Initialize basket position

    # Draw hand landmarks and calculate basket position
    if results.multi_hand_landmarks:
        x_coords = []
        y_coords = []
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the position of the wrist
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            x_coords.append(wrist.x * frame.shape[1])
            y_coords.append(wrist.y * frame.shape[0])
        
        # Calculate the average position if two hands are detected
        if len(x_coords) == 2:
            basket_x = int((x_coords[0] + x_coords[1]) / 2) - basket_width // 2
            basket_y = int((y_coords[0] + y_coords[1]) / 2) - basket_height
        else:
            basket_x = int(x_coords[0]) - basket_width // 2
            basket_y = int(y_coords[0]) - basket_height

        # Overlay the basket image instead of drawing a semicircle
        if basket_x is not None and basket_y is not None:
            overlay_basket(frame, basket_img, basket_x, basket_y)

    # Create new words
    if random.random() < 0.02:  
        words.append(create_word())

    # Update word positions and check for collisions
    for word in words[:]:
        word['y'] += 5  # Move word down
        
        # Render the word
        (text_width, text_height), _ = cv2.getTextSize(word['word'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(frame, word['word'], (word['x'], word['y'] + text_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Check for collision with basket
        if basket_x is not None and basket_y is not None:
            if (basket_x < word['x'] < basket_x + basket_width and
                basket_y < word['y'] < basket_y + 10):  # Only top of basket
                words.remove(word)
                score += 1

        # Remove words that have fallen off the screen
        if word['y'] > frame.shape[0]:
            words.remove(word)

    # Display score
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow('Game_1', frame)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
